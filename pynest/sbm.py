# stochastic block model

import numpy as np

# stochastic block model

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

class PPM(SBM):
    def __init__(self, p, lab_mat):

        # number of vertices
        num_vert = lab_mat.shape[0]

        # number of blocks
        num_bloc = lab_mat.shape[1]

        # q
        q = 1 - p

        bloc_prob_mat = q * np.ones((num_bloc, num_bloc)) + (p - q) * np.diag(np.ones(num_bloc))
        super().__init__(lab_mat, bloc_prob_mat)