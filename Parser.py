import csv
import os
from sklearn.decomposition import PCA

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

        for line in movies:
            for word in line.split(" "):
                x.append(word)

        print x


    def parse_ratings_data(self, filename):


        print 'Parsing ratings data'
        user_ids = []
        movie_ids = []
        ratings = []

        # Get a handle for the specified file
        data = open(filename)
        for line in data:
            print line
            arr = line.split()
            user_ids.append(arr[0])
            movie_ids.append(arr[1])
            ratings.append(arr[2])
        print user_ids

        return user_ids, movie_ids, ratings

if __name__ == '__main__':
    parser = Parser()
    parser.parse_movie_data('data/data.txt')
