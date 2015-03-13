from Parser import Parser

parser = Parser()
parser.parse_movie_data('data/movies.txt')
parser.get_dictionary_of_movie_types("Comedy", limit = 10)