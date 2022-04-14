import numpy as np

# random kernel graph


#========================================================================================
#========================================================================================

# M = (R^{n x d}, <.,.>) G = O_d (i.e. rdpg) 

# sample n points uniformly from first orthant of the unit d-ball
# variable names inspired by rejection sampling
def samp_unif_ball_orthant(n=1, d=1):
    count = 0
    samp_list = []
    while(count < n):
        y = np.random.uniform(size=d)
        if np.linalg.norm(y, ord=2) <= 1:
            samp_list.append(y)
            count = count + 1
        
    x = np.array(samp_list)
    
    return x

# get probability matrix
def get_prob_mat(x):
    p = x @ x.transpose()
    
    return p

# sample adjacency matrix
def samp_adj_mat(p):
    n = p.shape[0]
    ind = np.triu_indices(n, k=1)
    a_vals = np.random.binomial(1, p[ind])
    a = np.zeros((n,n))
    a[ind] = a_vals
    a = a.transpose()
    a[ind] = a_vals
    
    return a

# get spectral decomposition of self-adjoint M with eigenvals decreasing
def get_sd(M):
    lam, u = np.linalg.eig(M)
    ind = np.argsort(lam)[::-1]
    lam = lam[ind]
    u = u[:, ind]
    
    return lam, u

# get singular value decomposition of M with singular vals decreasing 
def get_svd(m, compute_uv=True):
    if compute_uv == True:
        u, s, vt = np.linalg.svd(m)
        ind = np.argsort(s)[::-1]
        u = u[:, ind]
        s = s[ind]
        vt = vt[ind, :]
    
        return u, s, vt
    else:
        s = np.linalg.svd(m, compute_uv=False)
        ind = np.argsort(s)[::-1]
        s = s[ind]
        
        return s

# calculate delta(M)
def get_delta(m):
    row_sums = np.sum(m, axis=1)
    max_row_sum = np.max(row_sums)
    
    return max_row_sum


# calculate gamma(M, d) with d in [0, ..., min(m,n) - 1] 
# where m, n are the number of rows, cols 
def get_gamma(m, d):
    s = get_svd(m, compute_uv=False)
    assert d < len(s) - 1
    sd = s[d]
    sdp = s[d+1]
    delta = get_delta(m)
    gamma = (sd - sdp) / delta
    
    return gamma
    
# get argmin of orthogonal matrix procrustes problem: find 
# orthogonal w s.t. |xw - y| is minimal
def get_opt_translate(x, y):
    u, s, vt = get_svd(x.transpose() @ y)
    w = u @ vt 
    
    return w

# get adjacency spectral embedding
def get_adj_spec_emb(a, d):
    a_sq = a.transpose() @ a
    lam_sq, u = get_sd(a_sq)
    
    lam_hat = (lam_sq[:d])**(1/4)
    u_hat = u[:, :d]
    x_hat = u_hat @ np.diag(lam_hat)
    
    return x_hat

# test stat from Tang et al
def get_test_stat(a, b, d):
    x_hat = get_adj_spec_emb(a, d)
    y_hat = get_adj_spec_emb(b, d)
    w = get_opt_translate(x_hat, y_hat)
    
    gam_a = get_gamma(a, d)
    gam_b = get_gamma(b, d)
    sa = np.sqrt(d / gam_a)
    sb = np.sqrt(d / gam_b)
    
    num = np.linalg.norm(x_hat @ w - y_hat, ord='fro')
    denom = sa + sb
    
    t = num / denom
    
    return t
#========================================================================================
#========================================================================================

# stochastic block model

# sample vertex label matrix
def samp_lab_mat(num_vert, lab_prob):
    lab_mat = np.random.multinomial(1, pvals=lab_prob, size=num_vert)

    return lab_mat

class SBM:
    def __init__(self, num_vert, lab_prob, dim_lat_pos):

        # label probabilities of length num_bloc
        self.lab_prob = lab_prob

        # number of blocks
        self.num_bloc = len(lab_prob)

        # label matrix of size num_vert x num_bloc
        self.lab_mat = samp_lab_mat(num_vert, lab_prob)

        # latent positions of size num_bloc x dim_lat_pos
        self.lat_pos = samp_unif_ball_orthant(self.num_bloc, dim_lat_pos)

        # block probability matrix of size num_bloc x num_bloc
        self.bloc_prob_mat = get_prob_mat(self.lat_pos)
        
        # probability matrix of size num_vert x num_vert
        self.prob_mat = self.lab_mat @ self.bloc_prob_mat @ self.lab_mat.transpose()
#========================================================================================
#========================================================================================

# distorted partition model

class DPM:
    def __init__(self, p, lab_mat, orth_pert, alpha):

        self.p = p

        self.lab_mat = lab_mat

        self.orth_pert = orth_pert

        self.alpha = alpha

        # number of vertices
        self.num_vert = lab_mat.shape[0]

        num_bloc = lab_mat.shape[1]        

        # q
        q = 1 - p

        # probability matrix of size num_bloc x num_bloc
        bloc_prob_mat = q * np.ones((num_bloc, num_bloc)) + (p - q) * np.diag(np.ones(num_bloc))
        
        # spectral decomposition of block probability matrix
        bloc_prob_spec = get_sd(bloc_prob_mat)
        self.bloc_prob_spec = bloc_prob_spec

        # square root of block probability matrix
        lam = bloc_prob_spec[0]
        U = bloc_prob_spec[1]
        Lam_sqrt = np.diag(lam)**.5

        # perturbed probability matrix of size num_vert x num_vert
        B = U @ Lam_sqrt
        Z = alpha * lab_mat @ B + (1 - alpha) * orth_pert

        self.prob_mat = Z @ Z.transpose()