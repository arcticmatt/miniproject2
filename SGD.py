from Parser import Parser
import random
import copy
import math
import numpy as np
import pdb
import time

class SGD:
    '''
    Runs SGD to learn a latent representation over users
    U and movies V. Note that in the code, U = U^T
    '''

    def __init__(self):
        # The number of latent factors. We will use 20 b/c Yisong Yue told us to.
        self.k = 20
        self.regularizer = 1
        self.learning_rate = .001
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

        parser = Parser()
        self.Y, self.training_points = parser.parse_ratings_data('data/data.txt')
        #self.Y = []
        #self.training_points = []
        #for i in range(10):
        #    row = []
        #    for j in range(10):
        #        row.append(j % 2)
        #    self.training_points.append((i, j))
        #    self.Y.append(row)

    def run(self):
        '''
        Run SGD until convergence.
        '''

        print 'Running SGD'
        epochs = 1
        self.old_error = 1000000
        self.should_stop = False
        while (True):
            print '=============== Epoch', epochs, '==============='

            # Keep track of old matrices to see how much this epoch changes them
            U_old = copy.deepcopy(self.U)
            V_old = copy.deepcopy(self.V)

            # Before each epoch, shuffle the training points to randomize the process.
            random.shuffle(self.training_points)
            count = 1
            for point in self.training_points:
                # Every 5000 points we will see if the error is going up; if so,
                # we will stop SGD
                if count % 5000 == 0:
                    print 'point #', count
                    error = self.get_error()
                    print 'Error =', error
                    print 'Old Error = ', self.old_error
                    if error > self.old_error:
                        print 'The error went up. Stopping!'
                        self.should_stop = True
                        break
                    self.old_error = error
                self.sgd_step(point)
                count += 1

            # Stop if error went up
            if self.should_stop:
                break

            # Get differences between updated and old matrices, filter out
            # all differences that are greater than .01
            U_diff = np.subtract(self.U, U_old)
            V_diff = np.subtract(self.V, V_old)
            high_diffs = []
            for U_row in U_diff:
                high_diffs.extend(filter(lambda x : x > self.cutoff, U_row))
            for V_row in V_diff:
                high_diffs.extend(filter(lambda x : x > self.cutoff, V_row))

            if not high_diffs:
                print 'Differences are all less than', self.cutoff, ', so we break!'
                break

            error = self.get_error()
            print 'Error =', error

            epochs += 1
            # Shrink learning rate
            self.learning_rate /= float(epochs)

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
        U_norm_grad_i = (self.regularizer / N) * U_i_row
        U_other_grad_i = -self.V[:,j] * ((self.Y[i][j] - self.Y_avg) - (np.dot(U_i_row, self.V[:,j]) + self.a[i] + self.b[j]))
        U_grads_i = self.learning_rate * (U_norm_grad_i + U_other_grad_i)
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
        Vt_norm_grad_j = (self.regularizer / N) * Vt_j_row
        Vt_other_grad_j = -self.U[i,:] * ((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        Vt_grads_j = self.learning_rate * (Vt_norm_grad_j + Vt_other_grad_j)
        Vt_grads = np.insert(Vt_grads, j, Vt_grads_j, 0)
        V_grads = np.transpose(Vt_grads)

        # Calculate the gradients for the a vector
        a_i_val = self.a[i]
        a_other_vals = np.append(self.a[:i], self.a[i + 1:])
        a_grads = self.learning_rate * (self.regularizer / N) * a_other_vals
        a_norm_grad_i = (self.regularizer / N) * a_i_val
        a_other_grad_i = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        a_grads_i = self.learning_rate * (a_norm_grad_i + a_other_grad_i)
        a_grads = np.insert(a_grads, i, a_grads_i)

        # Calculate the gradients for the b vector
        b_j_val = self.b[j]
        b_other_vals = np.append(self.b[:j], self.b[j + 1:])
        b_grads = self.learning_rate * (self.regularizer / N) * b_other_vals
        b_norm_grad_j = (self.regularizer / N) * b_j_val
        b_other_grad_j = -((self.Y[i][j] - self.Y_avg) - (np.dot(self.U[i,:], Vt_j_row) + self.a[i] + self.b[j]))
        b_grads_j = self.learning_rate * (b_norm_grad_j + b_other_grad_j)
        b_grads = np.insert(b_grads, j, b_grads_j)

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
            error = math.pow(self.Y[i][j] - UV[i][j], 2)
            sum_errors += error
        return sum_errors

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
