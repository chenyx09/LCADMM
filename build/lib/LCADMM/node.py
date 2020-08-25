import numpy as np
from scipy import interpolate, sparse
from numpy.linalg import norm
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
from itertools import combinations
import cvxpy as cp
from LCADMM.constants import *
solvers.options['show_progress'] = False


class ADMM_var():
    def __init__(self,dim,Ac = None,bc=None,lb = None, ub = None, x_des=None, x_norm = None):
        self.dim = dim
        self.NB = []
        self.ndx = {}
        self.ndx[self] = 0
        self.total_dim = dim
        self.p = ADMM_p0



        self.x_sol = None
        self.y1 = {}
        self.y2 = {}
        self.x_sol_old = None
        self.x_des = x_des
        self.opt_ready = False
        self.z = {}
        self.Q = None
        self.Qprox = None
        self.fprox = None
        self.f = None
        self.G = None
        self.h = None
        self.ns = 0
        self.A = np.empty([0,dim])
        self.b = np.empty(0)
        self.vardim = None
        self.N_nb = 0
        if x_norm is None:
            self.x_norm = np.ones(dim)
        else:
            self.x_norm = x_norm
        if not Ac is None:
            if Ac.shape[0]!=bc.shape[0]:
                raise ValueError('dimension mismatch')
            self.Ac = Ac
            self.bc = bc
        else:
            self.Ac = np.empty([0,self.dim])
            self.bc = np.empty(0)
        if not lb is None:
            for i in range(0,self.dim):
                if lb[i]>-np.inf:
                    Ai = np.zeros(self.dim)
                    Ai[i]=-1
                    self.Ac = np.vstack((self.Ac,Ai))
                    self.bc = np.append(self.bc,-lb[i])
        if not ub is None:
            for i in range(0,self.dim):
                if ub[i]<np.inf:
                    Ai = np.zeros(self.dim)
                    Ai[i]=1
                    self.Ac = np.vstack((self.Ac,Ai))
                    self.bc = np.append(self.bc,ub[i])


    def Add_ineq_cons(self,var1,A0,A1,b):

        if var1 is None:
            Anew = np.zeros([A0.shape[0],self.total_dim])
            Anew[:,self.ndx[self]:self.ndx[self]+self.dim] = A0
            self.A = np.vstack((self.A,Anew))
            self.b = np.append(self.b,b)
        elif isinstance(var1,ADMM_var):
            if var1 not in self.NB:
                self.NB.append(var1)
                self.ndx[var1]=self.total_dim
                self.total_dim +=var1.dim
                self.A = np.hstack((self.A,np.zeros([self.A.shape[0],var1.dim])))
            Anew = np.zeros([A0.shape[0],self.total_dim])
            Anew[:,self.ndx[self]:self.ndx[self]+self.dim] = A0
            Anew[:,self.ndx[var1]:self.ndx[var1]+var1.dim] = A1

            self.A = np.vstack((self.A,Anew))
            self.b = np.append(self.b,b)
        elif isinstance(var1,list) and isinstance(var1[0],ADMM_var):
            Anew = np.zeros([A0.shape[0],self.total_dim])
            Anew[:,self.ndx[self]:self.ndx[self]+self.dim] = A0
            for v,Av in zip(var1,A1):
                if v not in self.NB:
                    self.NB.append(v)
                    self.ndx[v]=self.total_dim
                    self.total_dim +=v.dim
                    self.A = np.hstack((self.A,np.zeros([self.A.shape[0],v.dim])))
                    Anew = np.hstack((Anew,np.zeros([Anew.shape[0],v.dim])))
                Anew[:,self.ndx[v]:self.ndx[v]+v.dim] = Av
            self.A = np.vstack((self.A,Anew))
            self.b = np.append(self.b,b)
        else:
            raise NotImplementedError

        self.N_nb = len(self.NB)
        self.opt_ready = False
    def generate_local_QP(self,beta):


        self.ns = self.A.shape[0]
        self.ndx['s'] = self.total_dim
        self.vardim = self.total_dim + self.ns
        self.G = np.empty([0,self.vardim])
        self.h = np.empty(0)

        Ai = np.zeros([self.Ac.shape[0],self.vardim])
        Ai[:,self.ndx[self]:self.ndx[self]+self.dim]=self.Ac
        self.G = np.vstack((self.G,Ai))
        self.h = np.append(self.h,self.bc)
        for var in self.NB:
            Ai = np.zeros([var.Ac.shape[0],self.vardim])
            Ai[:,self.ndx[var]:self.ndx[var]+var.dim]=var.Ac
            self.G = np.vstack((self.G,Ai))
            self.h = np.append(self.h,var.bc)

        As1 = np.zeros([self.ns,self.vardim])
        As1[:,self.ndx['s']:self.ndx['s']+self.ns]=-np.eye(self.ns)
        bs1 = np.zeros(self.ns)
        self.G = np.vstack((self.G,As1))
        self.h = np.append(self.h,bs1)
        As2 = np.zeros([self.ns,self.vardim])
        As2[:,self.ndx['s']:self.ndx['s']+self.ns]=-np.eye(self.ns)
        As2[:,0:self.total_dim] = self.A
        self.G = np.vstack((self.G,As2))
        self.h = np.append(self.h,self.b)

        if self.x_des is None:
            self.x_des = np.zeros(self.dim)

        self.Q = np.zeros([self.vardim,self.vardim])
        self.f = np.zeros(self.vardim)

        self.Q[self.ndx[self]:self.ndx[self]+self.dim,self.ndx[self]:self.ndx[self]+self.dim]=np.diag(1/self.x_norm**2)


        self.f[self.ndx[self]:self.ndx[self]+self.dim] += -self.x_des/self.x_norm**2

        self.f[self.ndx['s']:self.ndx['s']+self.ns] = beta*np.ones(self.ns)
        for var in self.NB:
            self.Q[self.ndx[var]:self.ndx[var]+var.dim,self.ndx[var]:self.ndx[var]+var.dim]=ADMM_RHO*np.diag(1/var.x_norm**2)
            if var not in self.z.keys():
                self.z[var]=np.zeros(var.dim)
            if var not in self.y1.keys():
                self.y1[var] = np.zeros(var.dim)
            if var not in self.y2.keys():
                self.y2[var] = np.zeros(self.dim)
            self.f[self.ndx[var]:self.ndx[var]+var.dim] += -ADMM_RHO*self.z[var]/var.x_norm**2+self.y1[var]/var.x_norm
            self.f[self.ndx[self]:self.ndx[self]+self.dim] -= self.y2[var]/self.x_norm

        if self.x_sol is None or self.x_sol.shape[0]!=self.total_dim:
            self.x_sol = np.zeros(self.total_dim)


        diag = np.zeros(self.vardim)
        diag[self.ndx[self]:self.ndx[self]+self.dim] = 1/(self.x_norm**2)
        for var in self.NB:
            diag[self.ndx[var]:self.ndx[var]+var.dim] = 1/(var.x_norm**2)


        self.Qprox = self.p*np.diag(diag)
        self.fprox = -self.p*diag*np.append(self.x_sol,np.zeros(self.ns))

        self.opt_ready = True

    def solve_local_QP(self,beta):
        if not self.opt_ready:
            self.generate_local_QP(beta)

        self.x_sol_old = self.x_sol
        self.f = np.zeros(self.vardim)
        self.f[self.ndx[self]:self.ndx[self]+self.dim] += -self.x_des/self.x_norm**2
        self.f[self.ndx['s']:self.ndx['s']+self.ns] = beta*np.ones(self.ns)
        for var in self.NB:
            self.f[self.ndx[var]:self.ndx[var]+var.dim] += -ADMM_RHO*self.z[var]/var.x_norm**2+self.y1[var]/var.x_norm
            self.f[self.ndx[self]:self.ndx[self]+self.dim] -= self.y2[var]/self.x_norm

        diag = np.zeros(self.vardim)
        diag[self.ndx[self]:self.ndx[self]+self.dim] = 1/(self.x_norm**2)
        for var in self.NB:
            diag[self.ndx[var]:self.ndx[var]+var.dim] = 1/(var.x_norm**2)


        self.Qprox = self.p*np.diag(diag)
        self.fprox = -self.p*diag*np.append(self.x_sol,np.zeros(self.ns))

        if (self.x_sol==0).all():
            self.Qprox = self.Qprox*0
            self.fprox = self.fprox*0

        Q = matrix(self.Q+self.Qprox)
        f = matrix(self.f+self.fprox)
        G = matrix(self.G)
        h = matrix(self.h)
        sol=solvers.qp(Q, f, G, h)
        self.x_sol = np.array(sol['x']).flatten()[0:self.total_dim]
    def reset(self,reset_dual = False):
        self.NB = []
        self.ndx = {}
        self.ndx[self] = 0
        self.total_dim = self.dim
        self.A = np.empty([0,self.dim])
        self.b = np.empty(0)
        self.Qprox = None

        if reset_dual:
            self.y1 = {}
            self.y2 = {}

            self.z = {}
            self.x_sol = None
        self.opt_ready = False
        self.Q = None
        self.f = None
        self.Qprox = None
        self.fprox = None
        self.p = ADMM_p0
        self.x_sol_old = None
        self.G = None
        self.h = None
        self.ns = 0
        self.vardim = None
        self.opt_ready = False
