# stochastic block model

import numpy as np

# stochastic block model

# sample vertex label matrix
def samp_lab_mat(num_vert, lab_prob):
    lab_mat = np.random.multinomial(1, pvals=lab_prob, size=num_vert)

    return lab_mat

class SBM:
    def __init__(self, lab_mat, bloc_prob_mat):

        # label matrix of size num_vert x num_bloc
        self.lab_mat = lab_mat

        # number of vertices
        self.num_vert = lab_mat.shape[0]

        # number of blocks
        self.num_bloc = lab_mat.shape[1]

        # block probability matrix of size num_bloc x num_bloc
        self.bloc_prob_mat = bloc_prob_mat
        
        # probability matrix of size num_vert x num_vert
        self.prob_mat = self.lab_mat @ self.bloc_prob_mat @ self.lab_mat.transpose()
#========================================================================================
#========================================================================================

class PPM(SBM):
    def __init__(self, p, lab_mat):

        self.p = p

        # number of vertices
        num_vert = lab_mat.shape[0]

        # number of blocks
        num_bloc = lab_mat.shape[1]

        # q
        q = 1 - p

        # probability matrix of size num_bloc x num_bloc
        bloc_prob_mat = q * np.ones((num_bloc, num_bloc)) + (p - q) * np.diag(np.ones(num_bloc))
        super().__init__(lab_mat, bloc_prob_mat)
#========================================================================================
#========================================================================================

