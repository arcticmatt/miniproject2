from Parser import Parser
import random
import copy
import numpy as np

class SGD:
    '''
    Runs SGD to learn a latent representation over users
    U and movies V. Note that in the code, U = U^T
    '''

    def __init__(self):
        # The number of latent factors. We will use 20 b/c Yisong Yue told us to.
        self.k = 20
        self.regularizer = 10
        self.learning_rate = .0001
        self.cutoff = .01

        # A  m x n  matrix of movie ratings, where y_ij corresponds to user (i+1)'s
        # rating for movie (j+1) (the plus-one's are because the matrix is 0-indexed)
        self.Y = None

        # Tuples of the form (user_id, movie_id). Used to index indices of the
        # Y matrix that actually hold data.
        self.training_points = []

        self.load_data() # Load the Y matrix

        Y_rows = len(self.Y)
        Y_cols = len(self.Y[0])

        # A  m x k  matrix. Initialized with random values.
        self.U = [[random.random() for i in range(self.k)] for i in range(Y_rows)]
        self.U = np.array(self.U) # make numpy array to make matrix operations easier

        # A  n x k  matrix. Initialized with random values.
        self.V = [[random.random() for i in range(Y_cols)] for i in range(self.k)]
        self.V = np.array(self.V) # make numpy array to make matrix operations easier

    def load_data(self):
        '''
        Load the Y matrix with values the
        file data/data.txt
        '''

        parser = Parser()
        self.Y, self.training_points = parser.parse_ratings_data('data/data.txt')

    def run(self):
        '''
        Run SGD until convergence.
        '''

        print 'Running SGD'
        num_epochs = 1
        while (True):
            print 'Epoch', num_epochs

            # Keep track of old matrices to see how much this epoch changes them
            U_old = copy.copy(self.U)
            V_old = copy.copy(self.V)

            # Before each epoch, shuffle the training points to randomize the process.
            random.shuffle(self.training_points)
            count = 1
            for point in self.training_points:
                #print 'point #', count
                self.sgd_step(point)
                count += 1

            # Get differences between updated and old matrices, filter out
            # all differences that are greater than .01
            U_diff = np.subtract(self.U, U_old)
            V_diff = np.subtract(self.V, V_old)
            print 'U_diff =', U_diff
            print 'V_diff =', V_diff
            low_diffs = []
            for U_row in U_diff:
                low_diffs.extend(filter(lambda x : x > self.cutoff, U_row))
            for V_row in V_diff:
                low_diffs.extend(filter(lambda x : x > self.cutoff, V_row))
            print 'low_diffs =', low_diffs

            if not low_diffs:
                print 'Differences are all less than .01, so we break!'
                break

            epochs += 1
            # Shrink learning rate
            self.learning_rate /= float(epochs)

        print 'Done running SGD'

    def sgd_step(self, point):
        '''
        Perform one step of stochastic gradient descent, given a tuple of
        the form (user_id, movie_id) which determines our y_ij in the gradient
        '''

        i, j = point
        U_rows = len(self.U)
        V_cols = len(self.V[0])
        U_grads = [0] * U_rows
        V_grads = [0] * V_cols

        # Loop through all rows in U and take the gradient of the target
        # function with respect to each one
        for k in range(0, U_rows):
            if k == i:
                U_grads[k] = self.learning_rate * ((self.regularizer / float(len(self.training_points))) * self.U[i,:] \
                        - self.V[:,j] * (self.Y[i][j] - np.multiply(self.U[i,:], self.V[:,j])))
            else:
                U_grads[k] = self.learning_rate * ((self.regularizer / float(len(self.training_points))) * self.U[k,:])

        # Loop through all columns in V and take the gradient of the target
        # function with respect to each one
        for k in range(0, V_cols):
            if k == i:
                V_grads[k] = self.learning_rate * ((self.regularizer / float(len(self.training_points))) * self.V[:,j] \
                        - self.U[i,:] * (self.Y[i][j] - np.multiply(self.U[i,:], self.V[:,j])))
            else:
                V_grads[k] = self.learning_rate * ((self.regularizer / float(len(self.training_points))) * self.V[:,k])

        # Turn gradient matrices into nparrays so we can subtract them easily from U and V
        U_grads = np.array(U_grads)
        V_grads = np.array(V_grads)
        V_grads = np.transpose(V_grads)

        # Perform shifts
        self.U = np.subtract(self.U, U_grads)
        self.V = np.subtract(self.V, V_grads)

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
