"""
This file contains the DAN and function to construct the neural networks
"""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as Mvn
import numpy as np 

class DAN(nn.Module):
    """
    A Data Assimilation Network class
    """
    def __init__(self, a_kwargs, b_kwargs, c_kwargs):

        nn.Module.__init__(self)
        self.a = Constructor(**a_kwargs)
        self.b = Constructor(**b_kwargs)
        self.c = Constructor(**c_kwargs)
        self.scores = {
            "RMSE_b": [],
            "RMSE_a": [],
            "LOGPDF_b": [],
            "LOGPDF_a": [],
            "LOSS": []}

    def forward(self, ha, x, y):
        """
        forward pass in the DAN
        """

        # cf slide 21 du cours
        # propagate past mem into prior mem
        hb = self.b.forward(ha)
        # translate prior mem into prior pdf
        pdf_b = self.c.forward(hb)
        # analyze prior mem
        ha = self.a.forward(torch.cat([hb, y], dim=1))
        # translate post mem into post pdf
        pdf_a = self.c.forward(ha)
        
        # (French ; English follows) les lignes ci-dessous reviennent a faire une estimation de Monte Carlo
        # de l'integrale parce que x, y est un echantillon de centaines de realisations
        # d'une trajectoire de x, y selon la loi jointe p(x,y)
        # donc faire la moyenne prise en l'echantillon revient a faire une estimation de Monte Carlo
        # de l'integrale suivant cette loi ; cf le cours il y a l'exemple pour L_0 calligraphique
        # Ici x et y sont x_t et y_t ; a chaque forward on travaille a un seul instant fixe
        # (English) the lines below are equivalent to a Monte Carlo estimation of the integral
        # because x, y is a sample of hundreds of realizations of a trajectory of x, y according to the joint
        # distribution p(x,y)
        Lt_qta = - torch.mean(pdf_a.log_prob(x))
        Lt_qtb = - torch.mean(pdf_b.log_prob(x))
        
        # rewrite loss 
        loss = Lt_qta + Lt_qtb
        
        # Compute scores
        with torch.no_grad():
            if Lt_qta is not None:
                self.scores["RMSE_b"].append(torch.mean(torch.norm(
                    pdf_b.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["RMSE_a"].append(torch.mean(torch.norm(
                    pdf_a.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["LOGPDF_b"].append(Lt_qtb.item())
                self.scores["LOGPDF_a"].append(Lt_qta.item())
                self.scores["LOSS"].append(loss.item())
                
        return loss, ha

    def clear_scores(self):
        """ clear the score lists
        """
        for v in self.scores.values():
            v.clear()

class Id(nn.Module):
    """ A simple id function
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """ trivial
        """
        return x

class Cst(nn.Module):
    """ A constant scale_vec
    """
    def __init__(self, init, dim=None):
        nn.Module.__init__(self)
        if isinstance(init, torch.Tensor):
            self.c = init.unsqueeze(0)
        else:
            raise NameError("Cst init unknown")

    def forward(self, x):
        return self.c.expand(x.size(0), self.c.size(0))

class Lin2d(nn.Module):
    # rotation dymnamics
    def __init__(self, x_dim, N, dt, init,
                 window=None):
        # dim de x: (mb,x_dim) (mb = number of "replicates")
        # N = nb of rotations to perform
        assert(x_dim == 2)
        nn.Module.__init__(self)
        
        theta = np.pi / 100.
        self.M = torch.empty(2, 2)
        sin = np.sin(theta)
        cos = np.cos(theta)
        self.M[0,0] = cos
        self.M[0,1] = sin
        self.M[1,0] = -sin
        self.M[1,1] = cos
        
        self.N = N
        
    def forward(self, x):
        # input x: (mb,x_dim) (mb = number of "replicates")
        # output Mx: (mb,x_dim)
        
        tM = self.M.T
        Mx = torch.mm(x, tM)
        for _ in range(self.N-1):
            Mx = torch.mm(Mx, tM)
        return Mx
    
class EDO(nn.Module):
    """ Integrates an EDO with RK4
    """
    def __init__(self, x_dim, N, dt, init,
                 window=None):
        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.N = N
        self.dt = dt
        if init == "95":
            """ Lorenz95 (96) initialization
            """
            self.window = (-2, -1, 0, 1)
            self.diameter = 4
            self.A = torch.tensor([[[0., 0., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 1., 0., 0.]]])
            self.b = torch.tensor([[0., 0., -1., 0.]])
            self.c = torch.tensor([8.])
        else:
            raise NameError("EDO init not available")

    def edo(self, x):
        # input x: (mb,x_dim)
        # output dx/dt: (mb,x_dim)
        # Hint: convert x into v (mb,x_dim,4), then reshape into (mb*x_dim,4)
        # and apply the matrix self.A using torch.nn.functional.bilinear, etc
        """v=
        x-2 x-1 x0 x1
        |   |   |  |
        """
        
        liste_x_permutes = torch.cat([torch.roll(x.unsqueeze(1), -i, 2) for i in self.window], 1)
        liste_x_permutes = torch.transpose(liste_x_permutes, 1, 2).reshape(-1, self.diameter)
        
        dtx = torch.nn.functional.bilinear(liste_x_permutes, liste_x_permutes, self.A) +\
            torch.nn.functional.linear(liste_x_permutes, self.b, self.c)

        return dtx.view(x.size(0), x.size(1))

    def forward(self, x):
        for _ in range(self.N):
            k1 = self.edo(x)
            k2 = self.edo(x + 0.5*self.dt*k1)
            k3 = self.edo(x + 0.5*self.dt*k2)
            k4 = self.edo(x + self.dt*k3)
            x = x + (self.dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        return x


class FullyConnected(nn.Module):
    """ Fully connected NN ending with a linear layer
    """
    def __init__(self, layers, activation_classname):
        nn.Module.__init__(self)
        n = len(layers)
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for
             d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList(
            [eval(activation_classname)() for _ in range(n-2)])

    def forward(self, h):
        for lin, act in zip(self.lins[:-1], self.acts):
            h = act(lin(h))
        return self.lins[-1](h)


class FcZero(nn.Module):
    """
    Fully connected neural network with ReZero trick
    """
    def __init__(self, dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        nn.Module.__init__(self)
        # init
        self.lins = nn.ModuleList([nn.Linear(dim, dim) for _ in range(deep)])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(deep)])
        self.l_alphas = [nn.Parameter(torch.Tensor([0.])) for _ in range(deep)]

    def forward(self, h):
        # rewrite output
        for lin, act, alpha in zip(self.lins, self.acts, self.l_alphas):
            h = h + alpha * act(lin(h))
        return h
        

class FcZeroLin(nn.Module):
    """
    FcZero network ending with linear layer
    """
    def __init__(self, in_dim, out_dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        nn.Module.__init__(self)
        # init
        self.lins = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(deep-1)] + [nn.Linear(in_dim, out_dim)])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(deep-1)])
        self.l_alphas = [nn.Parameter(torch.Tensor([0.])) for _ in range(deep-1)]

    def forward(self, h):
        for lin, act, alpha in zip(self.lins[:-1], self.acts, self.l_alphas):
            h = h + alpha * act(lin(h))
        # apply last (linear) layer
        h = self.lins[-1](h)
        return h

class Gaussian(Mvn):
    """
    Return a pytorch Gaussian pdf from args
    args is either a (loc, scale_tril) or a (x_dim, vec)
    """
    # "scale_tril" stands for "scale triangular (lower)"
    # scale_tril*scale_tril^T is the covariance matrix
    # The aim is to represent by a vector v both the mean and the covariance
    # that will be returned by the procoder c. Namely, the procoder c returns
    # W*ht + b = vt == (mu_t, Lambda_t) = (v[:x_dim], scale_tril)
    def __init__(self, *args):
        # in the subject (p. 2) we see that we do not compute the real 
        # Lambda_t matrix as scale_tril - instead, we modify its diagonal terms
        # so that they are neither too small nor to big
        self.minexp = torch.Tensor([-8.0])
        self.maxexp = torch.Tensor([8.0])
        
        if isinstance(args[0], int):
            """args is a (x_dim, vec)
            loc is the first x_dim coeff of vec
            if the rest is one coeff c then
                scale_tril = e^c*I
            else
                scale_tril is filled diagonal by diagonal
                starting by the main one
                (which is exponentiated to ensure strict positivity)
            """
            # Init Mvn by (x_dim, vec)
            x_dim, vec = args
            vec_dim = vec.size(-1)
            
            # Note that
            # vec.shape = (number of "replicates", dimension of the procoder)
            
            if vec_dim == x_dim + 1:
                # in this case, it means that vec is such that it only contains
                # enough values to fill the main diagonal, so we do:
                
                # Mvn by mean and sigma*I
                # scale_tril = e^c*I
                loc = vec[:, :x_dim]
                
                # torch.eye builds a (x_dim, x_dim) identity matrix
                # .unsqueeze(0) add a new dimension of size 1 at index "0" -> (1, x_dim, x_dim)
                scale_tril = torch.eye(x_dim)\
                                  .unsqueeze(0)\
                                  .expand(vec.size(0), -1, -1) # "expand" the size of first dimension
                                                               # ie each scale_tril(i, :, :) is an identity matrix
                                                               # for all i in 1...vec.size(0)
                                                               
                # we take the exponential
                scale_tril = torch.exp(vec[:, x_dim]).view(vec.size(0), 1, 1) * scale_tril
                # we prevent it to be either too big or too small
                scale_tril = torch.clamp(scale_tril, self.minexp, self.maxexp)
                                  
            else:
                # Mvn by mean and cov
                # rewrite scale_tril
                inds = self.vec_to_inds(x_dim, vec_dim)
                loc = vec[:, :x_dim]
                # see above to understand what follows
                scale_tril = torch.eye(x_dim)\
                                  .unsqueeze(0)\
                                  .expand(vec.size(0), -1, -1)
                scale_tril = torch.exp(vec[:, x_dim:2*x_dim]).view(vec.size(0), 1, x_dim) * scale_tril
                scale_tril = torch.clamp(scale_tril, self.minexp, self.maxexp)
                                  
                #scale_tril = scale_tril + torch.tril(vec[:, 2*x_dim:], diagonal=-1)
                # c'etait juste un essai rapide de la fonction "torch.tril", mais il aurait fallu adapter ca
                # pour repliquer ca le long du premier axe (celui des "repliquats")
                scale_tril[:, inds[0][x_dim:], inds[1][x_dim:]] = vec[:, 2*x_dim:]

            Mvn.__init__(self, loc=loc, scale_tril=scale_tril)

        else:
            """args is a loc, scale_tril
            """
            print("args is directly loc, scale_tril")
            Mvn.__init__(self, loc=args[0], scale_tril=args[1])

    def vec_to_inds(self, x_dim, vec_dim):
        """Computes the indices of scale_tril coeffs,
        scale_tril is filled main diagonal first

        x_dim: dimension of the random variable
        vec_dim: dimension of the vector containing
                 the coeffs of loc and scale_tril
        """
        ldiag, d, c = x_dim, 0, 0  # diag length, diag index, column index
        inds = [[], []]  # list of line and column indexes
        for i in range(vec_dim - x_dim):  # loop over the non-mean coeff
            inds[0].append(c+d)  # line index
            inds[1].append(c)  # column index
            if c == ldiag-1:  # the current diag end is reached
                ldiag += -1  # the diag length is decremented
                c = 0  # the column index is reinitialized
                d += 1  # the diag index is incremented
            else:  # otherwize, only the column index is incremented
                c += 1
        return inds


class Constructor(nn.Module):
    """Construct functions and conditional Gaussians from strings and kwargs
    - scale_vec_class is not None: return a Gaussian made from a vector,
        this vector is made of the concatenation of loc and scale_vec
    - scale_vec_class is None:
        if gauss_dim is not None: return a Gaussian made from a vector,
        else: return a vector
    """
    def __init__(self, loc_classname, loc_kwargs,
                 gauss_dim=None,
                 scale_vec_classname=None, scale_vec_kwargs=None):
        nn.Module.__init__(self)
        self.gauss_dim = gauss_dim
        self.loc = eval(loc_classname)(**loc_kwargs)
        if scale_vec_classname is not None:
            self.scale_vec =\
                eval(scale_vec_classname)(**scale_vec_kwargs)
        else:
            self.scale_vec = None

    def forward(self, *args):
        lc = self.loc(*args)
        if self.gauss_dim is not None:
            if self.scale_vec is not None:
                sc = self.scale_vec(*args)
                return Gaussian(self.gauss_dim, torch.cat((lc, sc), dim=1))
            else:
                return Gaussian(self.gauss_dim, lc)
        else:
            return lc
