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
            "Batman Forever" : 29,
            "Bad Boys" : 27,
            "Free Willy" : 78,
            "My Fair Lady" : 485,
            "Free Willy 2" : 35,
            "The Birdcage" : 25,
            "Nutty Professor" : 411,
            "GoldenEye" : 2,
            "Apollo 13" : 28,
            "Jurassic Park" : 82,
            "Forrest Gump" : 69,
            "Braveheart" : 22,
            "Twelve Monkeys" : 7,
            "Return of the Jedi" : 181,
            "Se7en" : 11,
            "Aliens" : 176,
            "Empire Strikes Back" : 172,
            "Star Wars" : 50,
            "2001 Space Odyssey" : 135,
            "Fargo" : 100,
            "Clockwork Orange" : 179,
        }

    def get_plot_data(self, data, ids):
        '''
        Filter out data points representing movies/users
        that do not match the provided id
        '''
        out = []
        for i in range(len(data[0])):
            if i in ids:
                out.append(map(lambda row: row[i], data))
        return transpose(out)

    def get_user_data(self, ids):
        return self.get_plot_data(self.U, ids)

    def get_movie_data(self, ids):
        return self.get_plot_data(self.V, ids)


    def __init__(self):
        self.sgd = SGD()
        self.sgd.run()
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
        try:
            self.U = (A2 * self.U).tolist()
            self.V = (A2 * self.V).tolist()
        except ValueError, e:
            print "Ran into an error, A2 dimensions: %s x %s."%(len(A2), len(A2[0]))
            print "U dimensions: %s x %s"%(len(self.U), len(self.U[0]))
            raise e


        print "New dimensions of U: %s x %s"%(len(self.U), len(self.U[0]))
        print "New dimensions of V: %s x %s"%(len(self.V), len(self.V[0]))


        # Get ids of movies we want to plot
        #to_plot = self.get_target_movies()

        # Plot horror movies
        to_plot = self.get_movie_type("Horror")

        # Get the coordinates of movies with the IDs we want to plot
        movie_data = self.get_movie_data(to_plot.values())

        # Plot movies we care about with labels
        self.plot(movie_data[0], movie_data[1], "$V_x$", "$V_y$", to_plot.keys())


    def plot(self, x, y, xlabel, ylabel, labels):
        '''
        Plots the specified x and y data lists
        using the specified x and y labels
        '''

        assert(len(x) == len(y) == len(labels))
        print "Plotting %s points: "%len(x)

        plt.scatter(x, y)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

        for label, x_val, y_val in zip(labels, x, y):

            textoffset = (-20, 20)

            # plt.annotate(
            #     label,
            #     xy = (x_val, y_val), xytext = textoffset,
            #     textcoords = 'offset points', ha = 'right', fontsize = 10, va = 'bottom',
            #     bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            #     arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            
            # Simpler plotting Style - KG
            plt.annotate(label, (x_val, y_val))

        plt.title ("2-D Approximation of Movie Data")
        plt.show()








if __name__ == '__main__':

    # Test matrix transposes
    test_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    tr = transpose(test_matrix)
    assert(tr == [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    # Run the visualizer
    v = Visualizer()
    v.run()


    '''
    matrix = [
    [1, 1, 1, 0, 0],
    [3, 3, 3, 0, 0],
    [4, 4, 4, 0, 0],
    [5, 5, 5, 0, 0],
    [0, 2, 0, 4, 4],
    [0, 0, 0, 5, 5],
    [0, 1, 0, 2, 2]
    ]

    # decomposer = d.TruncatedSVD(3)
    # result = decomposer.fit_transform(transpose(matrix))
    U, s, V = np.linalg.svd(matrix)

    print "U: %s"%U
    print "Sigma: %s"%s
    print "V: %s"%V
    # print map(lambda row: map(lambda col: round(col, 3), row), result)
    '''
