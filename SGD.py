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
        self.lmbda = 0
        self.regularizer = 10
        self.learning_rate = .01
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
        print self.Y_avg

        Y_rows = len(self.Y)
        Y_cols = len(self.Y[0])

        # A  m x k  matrix. Initialized with random values.
        self.U = [[random.random() for i in range(self.k)] for i in range(Y_rows)]
        self.U = np.array(self.U) # make numpy array to make matrix operations easier

        # A  k x n  matrix. Initialized with random values.
        self.V = [[random.random() for i in range(Y_cols)] for i in range(self.k)]
        self.V = np.array(self.V) # make numpy array to make matrix operations easier

        # Vector of bias/offset terms, one for each user
        self.a = [random.random() for i in range(Y_rows)]
        self.a = np.array(self.a)

        # Vector of bias/offset terms, one for each movie
        self.b = [random.random() for i in range(Y_cols)]
        self.b = np.array(self.b)

    def load_data(self):
        '''
        Load the Y matrix with values from the
        file data/data.txt
        '''

        self.parser = Parser()
        self.Y, self.training_points = self.parser.parse_ratings_data('data/data.txt')


    def update_nesterov_index(self):
        self.lmbda = (1.0 + math.sqrt(1.0 + 4.0 * (self.lmbda ** 2))) / 2.0
        self.gamma = (1.0 - self.lmbda) / (self.lmbda + 1.0)
        print "Setting gamma to %s"%self.gamma


    def run(self):
        '''
        Run SGD until convergence.
        '''

        print 'Running SGD'
        epochs = 1
        self.old_error = 1000000
        self.should_stop = False
        # Run a certain number of epochs; tweak based on trial and error
        while epochs < 50:
            self.update_nesterov_index()
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

        fname_U = 'U' + str(time.time()) + '_sav.txt'
        fname_V = 'V' + str(time.time()) + '_sav.txt'

        self.parser.write_to_csv(self.U, fname_U)
        self.parser.write_to_csv(self.V, fname_V)

        # Done with SGD; get error
        error = self.get_error()
        print 'Error =', error

    def sgd_step(self, point):
        '''
        Perform one step of stochastic gradient descent, given a tuple of
        the form (user_id, movie_id) which determines our y_ij in the gradient.
        Each step shifts
            - one row of the U matrix
            - one column of the V matrix
            - one value of the a vector
            - one value of the b vector
        '''

        i, j = point
        N = float(len(self.training_points))
        learning_rate = self.learning_rate * (1 - self.gamma)

        # Calculate the gradients for the U matrix. Do this by calculating the gradient
        # for every element in row i of U (remember that U is really U transpose, so this
        # corresponds to updating column i of the actual U matrix). We can speed this up
        # by using vector operations.
        U_i_row = self.U[i]
        U_i_reg = (self.regularizer / N) * self.U[i]
        U_i_error = -self.V[:,j] * ((self.Y[i][j] - self.Y_avg) - (np.dot(U_i_row, self.V[:,j]) + self.a[i] + self.b[j]))
        U_grads_i = learning_rate * (U_i_reg + U_i_error)
        # Shift the i'th row in our U matrix by the gradient vector we just calculated
        self.U[i] -= U_grads_i

        # Transpose V to make it easier to work with (so we can work with rows
        # instead of columns).
        # Calculuate the gradients for the V matrix. Do this by pulling out
        # the j'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the j'th row
        # separately and putting the results together. Then tranpose the results
        # to make the dimensions consistent with V.

        # Note that we transpose V to make it easier to work with (so we can work with rows
        # instead of columns).
        # Calculate the gradients for the V matrix. Do this by calculating the gradient
        # for every element in row j of V transpose. We can speed this up by using vector operations.
        Vt = np.transpose(self.V)
        Vt_j_row = Vt[j]
        Vt_j_error = -self.U[i,:] * ((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        Vt_j_regularization = (self.regularizer / N) * Vt_j_row
        V_j_grad = np.transpose( learning_rate * (Vt_j_error + Vt_j_regularization))
        # Shift the j'th column in our V matrix by the gradient vector we just calcullated
        self.V[:, j] -= V_j_grad

        # Calculate the gradient for the a vector
        a_regularization = self.a[i] * self.regularizer
        a_error = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        # Shift the i'th value in the a vector by the gradient we just calculated
        self.a[i] -= learning_rate * (a_error + a_regularization)

        # Calculate the gradient for the b vector
        b_regularization = self.b[j] * self.regularizer
        b_error = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        # Shift the j'th value in the b vector by the gradient we just calculated
        self.b[j] -= learning_rate * (b_error + b_regularization)

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
            error = math.pow(self.Y[i][j] - (self.Y_avg + UV[i][j] + self.a[i] + self.b[j]), 2)
            sum_errors += error
        return sum_errors

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
