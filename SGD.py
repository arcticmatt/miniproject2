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
        self.regularizer = 0.001
        self.learning_rate = 0.05
        self.cutoff = .0001
        self.lmbda = 0

        # A  m x n  matrix of movie ratings, where y_ij corresponds to user (i+1)'s
        # rating for movie (j+1) (the plus-one's are because the matrix is 0-indexed)
        self.Y = None

        # Tuples of the form (user_id, movie_id). Used to index indices of the
        # Y matrix that actually hold data.
        self.training_points = []

        self.load_data() # Load the Y matrix

        Y_rows = len(self.Y)
        Y_cols = len(self.Y[0])

        #print self.Y

        # A  m x k  matrix. Initialized with random values.
        self.U = [[random.random() for i in range(self.k)] for i in range(Y_rows)]
        self.U = np.array(self.U) # make numpy array to make matrix operations easier

        # A  k x n  matrix. Initialized with random values.
        self.V = [[random.random() for i in range(Y_cols)] for i in range(self.k)]
        self.V = np.array(self.V) # make numpy array to make matrix operations easier

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
        self.old_error = 10e9
        self.should_stop = False
        while (epochs < 20):
            print 'Epoch', epochs

            # Update coefficients for Nesterov GD
            self.update_nesterov_index()

            # Before each epoch, shuffle the training points to randomize the process.
            random.shuffle(self.training_points)
            count = 1
            for point in self.training_points:
                self.sgd_step(point)
                count += 1

            error = self.get_error()
            if error > self.old_error: 
                print 'The error went up. Stopping!'
                self.should_stop = True
                break

            self.old_error = error

            print 'Error =', error

            epochs += 1
            # Shrink learning rate
            self.learning_rate /= float(epochs)

        print 'Done running SGD'

        # Done with SGD; get error
        error = self.get_error()
        print 'Error =', error


    def update_nesterov_index(self):
        self.lmbda = (1.0 + math.sqrt(1.0 + 4.0 * (self.lmbda ** 2))) / 2.0
        self.gamma = (1.0 - self.lmbda) / (self.lmbda + 1.0)
        print "Setting gamma to %s"%self.gamma

    def sgd_step(self, point):
        '''
        Perform one step of stochastic gradient descent, given a tuple of
        the form (user_id, movie_id) which determines our y_ij in the gradient
        '''

        i, j = point
        N = float(len(self.training_points))

        learning_rate = self.learning_rate * self.lmbda

        # Calculuate the gradients for the U matrix. Do this by pulling out
        # the i'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the i'th row
        # separately and putting the results together.

        U_i_row = self.U[i]
        U_grads = learning_rate * (self.regularizer / N) * self.U
        U_other_grad_i = -self.V[:,j] * (self.Y[i][j] - (np.dot(U_i_row, self.V[:,j])))
        U_grads[i] += (learning_rate * U_other_grad_i)

        # Transpose V to make it easier to work with (so we can work with rows
        # instead of columns).
        # Calculuate the gradients for the V matrix. Do this by pulling out
        # the j'th row, calculating the gradient for the matrix sans that row (just
        # one multiplication), and then calculating the gradient for the j'th row
        # separately and putting the results together. Then tranpose the results
        # to make the dimensions consistent with V.

        Vt = np.transpose(self.V)
        Vt_j_row = Vt[j]
        Vt_grads = learning_rate * (self.regularizer / N) * Vt
        Vt_other_grad_j = -self.U[i,:] * (self.Y[i][j] - (np.dot(self.U[i,:], Vt_j_row)))
        Vt_grads[j] += (learning_rate * Vt_other_grad_j)
        V_grads = np.transpose(Vt_grads)
      

        self.U = np.subtract(self.U, U_grads)
        self.V = np.subtract(self.V, V_grads)
        # print "Done performing shift"
        # pdb.set_trace()

    def get_error(self):
        '''
        This method is called once we are done with SGD and returns the
        squared error between Y and UV.
        '''

        # Get squared differences between Y and UV for the indices of Y that
        # we initially initialized
        UV = np.dot(self.U, self.V)

        # print "%s\n"%self.U
        # print "%s\n"%self.V
        # print "\n\n\n"
        # print UV
        sum_errors = 0
        for point in self.training_points:
            i, j = point
            error = math.pow(self.Y[i][j] - UV[i][j], 2)
            sum_errors += error
        return sum_errors

if __name__ == '__main__':
    sgd = SGD()
    sgd.run()
