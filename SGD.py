from Parser import Parser
import random
import copy
import math
import numpy as np
import pdb
import time
import timeit

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
        self.cutoff = .0001

        # A  m x n  matrix of movie ratings, where y_ij corresponds to user (i+1)'s
        # rating for movie (j+1) (the plus-one's are because the matrix is 0-indexed)
        self.Y = None

        # Tuples of the form (user_id, movie_id). Used to index indices of the
        # Y matrix that actually hold data.
        self.training_points = []

        self.load_data() # Load the Y matrix

        # Average of all observations in Y
        self.Y_avg = (sum(map(sum, self.Y))) / float(len(self.training_points))

        Y_rows = len(self.Y)
        Y_cols = len(self.Y[0])

        # A  m x k  matrix. Initialized with random values.
        self.U = [[random.random() for i in range(self.k)] for i in range(Y_rows)]
        self.U = np.array(self.U) # make numpy array to make matrix operations easier

        # A  k x n  matrix. Initialized with random values.
        self.V = [[random.random() for i in range(Y_cols)] for i in range(self.k)]
        self.V = np.array(self.V) # make numpy array to make matrix operations easier

        # Vector of bias/offset terms, one for each user
        self.a = [random.random() / 1000 for i in range(Y_rows)]
        self.a = np.array(self.a)

        # Vector of bias/offset terms, one for each movie
        self.b = [random.random() / 1000 for i in range(Y_cols)]
        self.b = np.array(self.b)

    def load_data(self):
        '''
        Load the Y matrix with values from the
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
        self.old_error = 1000000
        self.should_stop = False
        while epochs < 50:
            print '=============== Epoch', epochs, '==============='

            # Keep track of old matrices to see how much this epoch changes them
            # U_old = copy.deepcopy(self.U)
            # V_old = copy.deepcopy(self.V)

            # Before each epoch, shuffle the training points to randomize the process.
            random.shuffle(self.training_points)
            count = 1
            for point in self.training_points:
                self.sgd_step(point)
                count += 1

            # Stop if error went up
            if self.should_stop:
                break

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
        U_grads = self.learning_rate * (self.regularizer / N) * self.U
        U_other_grad_i = -self.V[:,j] * ((self.Y[i][j] - self.Y_avg) - (np.dot(U_i_row, self.V[:,j]) + self.a[i] + self.b[j]))
        U_grads[i] += (self.learning_rate * U_other_grad_i)

        # Transpose V to make it easier to work with (so we can work with rows
        # instead of columns).
        # Calculuate the gradients for the V matrix. Do this by pulling out
        # the j'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the j'th row
        # separately and putting the results together. Then tranpose the results
        # to make the dimensions consistent with V.

        Vt = np.transpose(self.V)
        Vt_j_row = Vt[j]
        Vt_grads = self.learning_rate * (self.regularizer / N) * Vt
        Vt_other_grad_j = -self.U[i,:] * ((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        Vt_grads[j] += (self.learning_rate * Vt_other_grad_j)
        V_grads = np.transpose(Vt_grads)


        # Calculate the gradients for the a vector
        a_i_val = self.a[i]
        a_grads = self.learning_rate * (self.regularizer / N) * self.a
        a_other_grad_i = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        a_grads[i] += self.learning_rate * a_other_grad_i        

        b_j_val = self.b[j]
        b_grads = self.learning_rate * (self.regularizer / N) * self.b
        b_other_grad_j = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        a_grads[i] += self.learning_rate * b_other_grad_j

        # Perform shifts
        self.U = np.subtract(self.U, U_grads)
        self.V = np.subtract(self.V, V_grads)
        self.a = np.subtract(self.a, a_grads)
        self.b = np.subtract(self.b, b_grads)

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
            error = math.pow(self.Y[i][j] - (UV[i][j] + self.a[i] + self.b[j]), 2)
            sum_errors += error
        return sum_errors

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
