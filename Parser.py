import csv
import os

class Parser:
    '''
    Parses a tab/space seperated data file
    '''
    def __init__(self):
        pass

    def parse_movie_data(self, filename):
        # TODO

        print 'Parsing movie data'
        x = []
        y = []

        # Get a handle for the specified file
        movies = open(filename)

        for line in movies:
            for word in line.split(" "):
                x.append(word)


    def parse_ratings_data(self, filename):
        '''Parses the ratings data, which consists of:
        a user id, a movie id, a rating
        '''

        print 'Parsing ratings data'
        user_ids = []
        movie_ids = []
        ratings = []

        # Get a handle for the specified file
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
            user_id = user_ids[i] - 1 # decrement by 1 because matrix is 0 indexed
            movie_id = movie_ids[i] - 1 # decrement by 1 because matrix is 0 indexed
            rating = ratings[i]
            Y[user_id][movie_id] = rating

        return Y


if __name__ == '__main__':
    parser = Parser()
    data = parser.parse_ratings_data('data/data.txt')
    print data
