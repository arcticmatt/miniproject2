from Parser import Parser
import random
import copy
import numpy as np
from SGD import SGD
from sklearn import decomposition as d
import matplotlib.pyplot as plt


def transpose(matrix):
    out = []
    for i in range(len(matrix[0])):
        out.append(map(lambda row: row[i], matrix))
    return out


class Visualizer:
    '''
    Runs SVD to decompose the data matrix X into U Sigma V^T, project X
    onto the first k columns of U, and visualize the resulting data set

    '''

    def get_movie_type(self, type):
        self.parser = Parser()
        self.parser.parse_movie_data('data/movies.txt')
        return self.parser.get_dictionary_of_movie_types(type, limit = 20)

    def get_target_movies(self):
        '''
        Return a dict mapping the strings (the names of movies
        possessing IDs we want to plot) to integers
        (the movies' IDs)
        '''

        return {
            "Batman Forever" : 28,
            "Batman Returns" : 230,
            "Batman & Robin" : 253,
            "Batman" : 402,
            "Bad Boys" : 26,
            "Free Willy" : 77,
            "My Fair Lady" : 484,
            "Free Willy 2" : 34,
            "The Birdcage" : 24,
            "Nutty Professor" : 410,
            "GoldenEye" : 1,
            "Apollo 13" : 27,
            "Jurassic Park" : 81,
            #"Forrest Gump" : 68,
            "Braveheart" : 21,
            #"Twelve Monkeys" : 6,
            "Return of the Jedi" : 180,
            "Se7en" : 10,
            "Aliens" : 175,
            "Empire Strikes Back" : 171,
            "Star Wars" : 49,
            "2001 Space Odyssey" : 134,
            #"Fargo" : 99,
            #"Clockwork Orange" : 178,
            "Falling in Love Again (1980)": 1374,
            "Love Affair": 1297,
            #"When A MAn Loves a Woman": 1221,
            #"What's Love Got to Do With it": 942
            "Addicted to Love": 535,
            #"The Love Bug": 139
        }

    def get_plot_data(self, data, ids):
        '''
        Filter out data points representing movies/users
        that do not match the provided id
        '''

        out = []
        if ids == None:
            return data
        for i in ids:
            out.append(map(lambda row: row[i], data))
        return transpose(out)

    def get_user_data(self, ids=None):
        return self.get_plot_data(self.U, ids)

    def get_movie_data(self, ids=None):
        print len(self.V)
        return self.get_plot_data(self.V, ids)

    def visualize_without_running(self, U_fname, V_fname):
        self.parser = Parser()
        self.U = transpose(self.parser.read_from_csv(U_fname, 'U'))
        self.V = self.parser.read_from_csv(V_fname, 'V')

    def __init__(self, norun = False, uname = 'U.txt', vname = 'V.txt'):
        if norun:
            self.visualize_without_running(uname, vname)
        else:
            self.sgd = SGD()
            try:
                self.sgd.run()
            except KeyboardInterrupt:
                pass

            self.U = transpose(self.sgd.U)
            self.V = self.sgd.V

    def run(self, num_components=2):
        print "Original dimensions of U: %s x %s"%(len(self.U), len(self.U[0]))
        print "Original dimensions of V: %s x %s"%(len(self.V), len(self.V[0]))

        # Perform an SVD on U = A * sigma * B
        A, sigma, B = np.linalg.svd(self.V)

        # Extract the first two columns of A
        A2 = np.matrix(transpose(map(lambda row: row[:2], A)))

        # Project U and V to the first two columns of A
        self.U = (A2 * self.U).tolist()
        self.V = (A2 * self.V).tolist()


        print "New dimensions of U: %s x %s"%(len(self.U), len(self.U[0]))
        print "New dimensions of V: %s x %s"%(len(self.V), len(self.V[0]))


        # Get info and coordinates for 'target' movies (those used by
        # Prof. Yue in his example)
        target_movies_info = self.get_target_movies()
        movie_data = self.get_movie_data(target_movies_info.values())
        target_movie_names = target_movies_info.keys()

        print movie_data
        print target_movie_names

        # Get info and coordinates for musical movies
        musical_movie_info = self.get_movie_type("Musical")
        musical_movie_names = musical_movie_info.keys()
        musical_movie_data = self.get_movie_data(musical_movie_info.values())

        # Get info and coordinates for all users and movies
        all_movies = self.get_movie_data()
        print all_movies
        user_data = self.get_user_data()

        # Plot movies we care about with labels
        data_series = [(movie_data[0], movie_data[1], target_movie_names)]
        self.plot(data_series, "$V_x$", "$V_y$", "2-D Approximation of Movie Data")

        print 'all movies: ' + str(len(all_movies))
        print len(all_movies[0])

        # Plot all movies and users
        data_series = [(all_movies[0], all_movies[1], []), (user_data[0], user_data[1], [])]
        self.plot(data_series, "$V_x$", "$V_y$", "All User (red) and Movie (blue) Data")

        # Plot all musical Movies
        data_series = [(musical_movie_data[0], musical_movie_data[1], musical_movie_names)]
        self.plot(data_series, "$V_x$", "$V_y$", "2-D Approximation of musical Movie Data")

        # Plot all movies, and musical movies
        data_series = [(all_movies[0], all_movies[1], []), (musical_movie_data[0], musical_movie_data[1], [])]
        self.plot(data_series, "$V_x$", "$V_y$", "All Movies (blue) and musical (Red) Data")

    def plot(self, data_series, xlabel, ylabel, title):
        '''
        Plots the specified x and y data lists
        using the specified x and y labels
        '''

        colors = ["orange", "green", "red", "blue"]

        for x, y, labels in data_series:
            plt.scatter(x, y, color=colors.pop())
            plt.scatter([np.mean(x)], [np.mean(y)], color = colors.pop())
            print np.mean(x)
            print np.mean(y)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)

            for label, x_val, y_val in zip(labels, x, y):

                textoffset = (-20, 20)

                plt.annotate(
                        label,
                        xy = (x_val, y_val), xytext = textoffset,
                        textcoords = 'offset points', ha = 'right', fontsize = 10, va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

                # Simpler plotting Style - KG
                #plt.annotate(label, (x_val, y_val))
        plt.title (title)
        plt.show()


if __name__ == '__main__':

    # Test matrix transposes
    test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tr = transpose(test_matrix)
    assert(tr == [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    # Run the visualizer
    v = Visualizer(norun = True, uname = 'U1426279119.57_sav.txt', vname = 'V1426279119.57_sav.txt')
    #v = Visualizer()
    v.run()
