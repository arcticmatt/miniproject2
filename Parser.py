import csv
import os
import numpy as np
import ast

class Parser:
    '''
    Parses a tab/space seperated data file
    '''
    def __init__(self):
        pass

    def parse_movie_data(self, filename):

        print 'Parsing movie data'
        x = []
        y = []

        # Get a handle for the specified file
        movies = open(filename)

        # We need this list of genres in case the caller
        # asks for a specific genre, see get_dictionary_of_movie_types
        self.genres = ["movieId", "movieTitle", "Unkown", 
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


        self.movies_arr = []
        for line in movies:
            line = line.rstrip()
            movie = []
            for word in line.split('\t'):
                movie.append(word)
            self.movies_arr.append(movie)

    '''
    We will use this method to write U and V to a CSV.
    '''
    def write_to_csv (self, matrix, filename):
        path = 'savedresults/' + filename
        print 'Saving ' + filename
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'w+') as sub_file:
            file_writer = csv.writer(sub_file)
            for i in range(0, len(matrix)):
                file_writer.writerow(matrix[i])

    '''
    We will use this method to read U and V to a CSV.
    '''
    def read_from_csv(self, filename, matrix_name = 'U'):
        if matrix_name != 'U' and matrix_name != 'V':
            raise ValueError ("We can only fetch U or V.")

        path = 'savedresults/' + filename
        print 'Reading saved matrix ' + matrix_name

        matrix = []

        if os.path.exists(path):
            with open(path, 'rb') as csvfile:
                file_reader = csv.reader(csvfile)
                for row in file_reader:
                    row = map(float, row)
                    matrix.append(row)

        return np.array(matrix)




    def get_dictionary_of_movie_types(self, type, limit = 100):
        movies = {}
        index_to_look_for = -1

        for i in range(len(self.genres)):
            if self.genres[i] == type:
                index_to_look_for = i

        # Either we didn't find that type of genre,
        # or you're looking for 'movieId' or 'movieTitle',
        # which aren't valid things to look for.
        if index_to_look_for < 2:
            raise ValueError ("The type you passed is probably not valid...")

        for movie in self.movies_arr:
            if movie[index_to_look_for] == '1' and len(movies) < limit:
                movies[movie[1]] = int(movie[0])

        print movies
        return movies


    
    def get_horror_movies(self):
        horror_films = {}
        for movie in self.movies_arr:
            if movie[13] == '1':
                horror_films[movie[1]] = int(movie[0])

        return horror_films



    def parse_ratings_data(self, filename):
        '''Parses the ratings data, which consists of:
        a user id, a movie id, a rating

        Returns a tuple consisting of the resulting Y matrix and
        a list of training_points that tell us which indices of the Y matrix
        contain actual data
        '''

        print 'Parsing ratings data'
        user_ids = []
        movie_ids = []
        ratings = []

        # Tuples of the form (user_id, movie_id). Used to index indices of the
        # Y matrix that actually hold data.
        training_points = []

        # Get a handle for the specified file.
        data = open(filename)
        for line in data:
            line = line.rstrip()
            arr = map(int, line.split('\t'))
            user_ids.append(arr[0])
            movie_ids.append(arr[1])
            ratings.append(arr[2])

        num_rows = max(user_ids)
        num_columns = max(movie_ids)
        Y = [[0 for i in range(num_columns)] for i in range(num_rows)]
        for i in range(0, len(user_ids)):
            user_id = user_ids[i] - 1 # Decrement by 1 because matrix is 0 indexed.
            movie_id = movie_ids[i] - 1 # Decrement by 1 because matrix is 0 indexed.
            training_points.append((user_id, movie_id))
            rating = ratings[i]
            Y[user_id][movie_id] = rating

        return Y, training_points


if __name__ == '__main__':
    parser = Parser()
    data = parser.parse_ratings_data('data/data.txt')
    print data
