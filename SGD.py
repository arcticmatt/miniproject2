from Parser import Parser
import random
import copy
import math
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
        self.learning_rate = .001
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
        epochs = 1
        while (True):
            print 'Epoch', epochs

            # Keep track of old matrices to see how much this epoch changes them
            U_old = copy.copy(self.U)
            V_old = copy.copy(self.V)

            # Before each epoch, shuffle the training points to randomize the process.
            random.shuffle(self.training_points)
            count = 1
            for point in self.training_points:
                if count % 5000 == 0:
                    print 'point #', count
                    #error = self.get_error()
                    #print 'Error =', error
                self.sgd_step(point)
                count += 1

            # Get differences between updated and old matrices, filter out
            # all differences that are greater than .01
            U_diff = np.subtract(self.U, U_old)
            V_diff = np.subtract(self.V, V_old)
            high_diffs = []
            for U_row in U_diff:
                high_diffs.extend(filter(lambda x : x > self.cutoff, U_row))
            for V_row in V_diff:
                high_diffs.extend(filter(lambda x : x > self.cutoff, V_row))

            #if not high_diffs:
                #print 'Differences are all less than .01, so we break!'
                #break

            error = self.get_error()
            print 'Error =', error

            epochs += 1
            # Shrink learning rate
            #self.learning_rate /= float(epochs)

        print 'Done running SGD'

        # Done with SGD; get error
        error = self.get_error()
        print 'Error =', error

    def sgd_step(self, point):
        '''
        Perform one step of stochastic gradient descent, given a tuple of
        the form (user_id, movie_id) which determines our y_ij in the gradient
        '''

        i, j = point
        N = float(len(self.training_points))

        # Calculuate the gradients for the U matrix. Do this by pulling out
        # the i'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the i'th row
        # separately and putting the results together.
        U_i_row = self.U[i]
        U_other_rows = np.vstack((self.U[:i], self.U[i + 1:]))
        U_grads = self.learning_rate * (self.regularizer / N) * U_other_rows
        U_grads_i = self.learning_rate * ((self.regularizer / N) * U_i_row \
                        - self.V[:,j] * (self.Y[i][j] - np.multiply(U_i_row, self.V[:,j])))
        U_grads = np.insert(U_grads, i, U_grads_i, 0)

        # Transpose V to make it easier to work with (so we can work with rows
        # instead of columns).
        # Calculuate the gradients for the V matrix. Do this by pulling out
        # the j'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the j'th row
        # separately and putting the results together. Then tranpose the results
        # to make the dimensions consistent with V.
        Vt = np.transpose(self.V)
        Vt_j_row = Vt[j]
        Vt_other_rows = np.vstack((Vt[:j], Vt[j + 1:]))
        Vt_grads = self.learning_rate * (self.regularizer / N) * Vt_other_rows
        Vt_grads_j = self.learning_rate * ((self.regularizer / N) * Vt_j_row \
                        - self.U[i,:] * (self.Y[i][j] - np.multiply(self.U[i,:], Vt_j_row)))
        Vt_grads = np.insert(Vt_grads, j, Vt_grads_j, 0)
        V_grads = np.transpose(Vt_grads)

        # Perform shifts
        self.U = np.add(self.U, U_grads)
        self.V = np.add(self.V, V_grads)

    def get_error(self):
        '''
        This method is called once we are done with SGD and returns the
        squared error between Y and UV.
        '''

        # Get squared differences between Y and UV for the indices of Y that
        # we initially initialized
        UV = np.dot(self.U, self.V)
        sum_errors = 0
        for point in self.training_points:
            i, j = point
            error = math.pow(self.Y[i][j] - UV[i][j], 2)
            sum_errors += error
        return sum_errors

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
