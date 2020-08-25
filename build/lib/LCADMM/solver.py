import numpy as np
from scipy import interpolate, sparse
from numpy.linalg import norm
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
from itertools import combinations
import cvxpy as cp
from LCADMM.node import ADMM_var
from LCADMM.constants import *
import pdb

class ADMM_problem():
    def __init__(self,beta = ADMM_BETA):
        self.N = 0
        self.vars = []
        self.ndx = {}
        self.vardim = 0
        self.x_sol = None
        self.beta = ADMM_BETA
    def add_agent(self,agent):
        self.N+=1
        self.vars.append(agent)
        self.ndx[agent]=self.vardim
        self.vardim+=agent.dim
    def Add_ineq_cons(self,vars,As,b):
        if isinstance(vars,ADMM_var):
            if vars not in self.vars:
                self.add_agent(vars)
            vars.Add_ineq_cons(None,As,None,b)
        elif isinstance(vars,list):
            M = len(vars)
            for Ai in As:
                Ai = Ai/M
            b = b/M
            # pdb.set_trace()
            for var in vars:
                if var not in self.vars:
                    self.add_agent(var)
            if M==1:
                vars[0].Add_ineq_cons(None,As[0],None,b)
            elif M==2:
                vars[0].Add_ineq_cons(vars[1],As[0],As[1],b)
                vars[1].Add_ineq_cons(vars[0],As[1],As[0],b)
            else:
                for i in range(0,len(vars)):
                    subvar = vars[0:i]+vars[i+1:]
                    subAs = As[0:i]+As[i+1:]
                    var.Add_ineq_cons(subvar,As[i],subAs,b)

    def solve(self,maxiter = 1e6, tol = 1e-4):
        iter = 0
        err = 1.
        for var in self.vars:
            var.solve_local_QP(self.beta)
        # pdb.set_trace()
        while iter<maxiter and err>tol:
            err = 0.
            iter +=1
            # print(self.vars[0].y1[self.vars[0].NB[0]])
            # print(self.vars[0].x_sol[0:2])
            # print(self.vars[0].f[0:2])
            for var in self.vars:

                if ADMM_ADAPTIVE:
                    delta_x = var.x_sol[0:var.total_dim]-var.x_sol_old
                    hh = 0
                    hh +=(var.p+ADMM_RHO*var.N_nb)*norm(delta_x)**2
                    du = hh
                for var1 in var.NB:

                    dx = (var.x_sol[var.ndx[var1]:var.ndx[var1]+var1.dim]-var1.x_sol[var1.ndx[var1]:var1.ndx[var1]+var1.dim]).flatten()
                    if ADMM_ADAPTIVE:
                        hh+=norm(dx)**2*(2-ADMM_GAMMA)*ADMM_RHO+2*ADMM_RHO*dx.dot(delta_x[var.ndx[var1]:var.ndx[var1]+var1.dim])
                        du +=ADMM_RHO*ADMM_GAMMA*norm(dx)**2
                    var.y1[var1]+=ADMM_GAMMA*ADMM_RHO*dx/var1.x_norm
                    var1.y2[var] = var.y1[var1]
                    var1.z[var] = var.x_sol[var.ndx[var]:var.ndx[var]+var.dim].flatten()

                    err = max(err,norm(dx/var1.x_norm))
                    # if norm(dx/var1.x_norm)>1.5:
                    #     print(var.y1[var1])
                    # if iter>60 and norm(dx/var1.x_norm)>1.9:
                    #     pdb.set_trace()
                if ADMM_ADAPTIVE:
                    if hh<0.5*du:
                        var.p+=ADMM_q
            if err<=tol:
                break
            print(err)
            for var in self.vars:
                var.solve_local_QP(self.beta)
        # pdb.set_trace()
        self.x_sol = np.zeros(self.vardim)
        for var in self.vars:
            self.x_sol[self.ndx[var]:self.ndx[var]+var.dim]= var.x_sol[var.ndx[var]:var.ndx[var]+var.dim].flatten()
        return self.x_sol
    def solve_centralized(self):
        ndx = {}
        dim = 0
        for var in self.vars:
            ndx[var] = dim
            dim+=var.dim
        Ac = np.empty([0,dim])
        bc = np.empty(0)
        for var in self.vars:
            Aci = np.zeros([var.Ac.shape[0],dim])
            bci = var.bc
            Aci[:,ndx[var]:ndx[var]+var.dim] = var.Ac
            Ac = np.vstack((Ac,Aci))
            bc = np.append(bc,bci)
        Ain = np.empty([0,dim])
        bin = np.empty(0)
        pairs = combinations(self.vars,2)
        for pair in pairs:
            var0 = pair[0]
            var1 = pair[1]
            if var1 in var0.NB:
                idx = np.where((var0.A[:,var0.ndx[var1]:var0.ndx[var1]+var1.dim]!=0).any(axis=1))[0]
                Anew = np.zeros([idx.shape[0],dim])
                Anew[:,ndx[var0]:ndx[var0]+var0.dim] = var0.A[idx,var0.ndx[var0]:var0.ndx[var0]+var0.dim]
                Anew[:,ndx[var1]:ndx[var1]+var1.dim] = var0.A[idx,var0.ndx[var1]:var0.ndx[var1]+var1.dim]
                bnew = var0.b[idx]
                Ain = np.vstack((Ain,Anew))
                bin = np.append(bin,bnew)

        ns = Ain.shape[0]
        Ac = np.hstack((Ac,np.zeros([Ac.shape[0],ns])))
        Ain = np.hstack((Ain,-np.eye(ns)))
        Ain1 = np.hstack((np.zeros([ns,dim]),-np.eye(ns)))
        Ain = np.vstack((Ain,Ain1))
        bin = np.append(bin,np.zeros(ns))

        Q = np.zeros([dim+ns,dim+ns])
        for var in self.vars:
            Q[ndx[var]:ndx[var]+var.dim,ndx[var]:ndx[var]+var.dim]=np.diag(1/var.x_norm**2)
        f = np.zeros(dim+ns)
        for var in self.vars:
            f[ndx[var]:ndx[var]+var.dim]=-var.x_des/(var.x_norm**2)
        f[dim:dim+ns] = 2*self.beta

        G = np.vstack((Ac,Ain))
        h = np.append(bc,bin)
        sol=solvers.qp(matrix(Q), matrix(f), matrix(G), matrix(h))
        print(sol['primal objective'])
        x_sol = np.array(sol['x']).flatten()[0:dim]
        return x_sol

    def solve_constrained_centralized(self,exp_relax = False):
        if not exp_relax:
            ndx = {}
            con_ndx = {}
            dim = 0
            Q = np.empty([0,0])
            f = np.empty(0)
            G = np.empty([0,0])
            h = np.empty(0)
            for var in self.vars:

                if not var.opt_ready:
                    var.generate_local_QP(self.beta)
                ndx[var] = dim
                dim += var.vardim
                var.Q = np.zeros([var.vardim,var.vardim])
                var.f = np.zeros(var.vardim)

                var.Q[var.ndx[var]:var.ndx[var]+var.dim,var.ndx[var]:var.ndx[var]+var.dim]=np.diag(1/var.x_norm**2)


                var.f[var.ndx[var]:var.ndx[var]+var.dim] += -var.x_des/(var.x_norm**2)
                var.f[var.ndx['s']:var.ndx['s']+var.ns] = self.beta*np.ones(var.ns)

                Q = block_diag(Q,var.Q)
                f = np.append(f,var.f)
                G = block_diag(G,var.G)
                h = np.append(h,var.h)

            vardim = dim
            # A = np.empty([0,vardim])
            # b = np.empty(0)
            condim = G.shape[0]

            for var in self.vars:
                for var1 in var.NB:
                    con_ndx[(var, var1)] = condim
                    Anew = np.zeros([var1.dim*2,vardim])
                    Anew[0:var1.dim,ndx[var]+var.ndx[var1]:ndx[var]+var.ndx[var1]+var1.dim] = np.eye(var1.dim)
                    Anew[0:var1.dim,ndx[var1]+var1.ndx[var1]:ndx[var1]+var1.ndx[var1]+var1.dim] = -np.eye(var1.dim)

                    Anew[var1.dim:2*var1.dim,ndx[var]+var.ndx[var1]:ndx[var]+var.ndx[var1]+var1.dim] = -np.eye(var1.dim)
                    Anew[var1.dim:2*var1.dim,ndx[var1]+var1.ndx[var1]:ndx[var1]+var1.ndx[var1]+var1.dim] = np.eye(var1.dim)

                    G = np.vstack((G,Anew))
                    h = np.append(h,np.zeros(2*var1.dim))

                    condim = G.shape[0]
            sol=solvers.qp(matrix(Q), matrix(f), matrix(G), matrix(h))
            print(sol['primal objective'])
            sol_x = np.array(sol['x']).flatten()
            sol_z = np.array(sol['z']).flatten()
            x_sol = np.empty(0)
            y_sol = {}
            for var in self.vars:
                x_sol = np.append(x_sol,sol_x[ndx[var]+var.ndx[var]:ndx[var]+var.ndx[var]+var.dim])
                for var1 in var.NB:
                    y1 = sol_z[con_ndx[(var,var1)]:con_ndx[(var,var1)]+var1.dim]
                    y2 = sol_z[con_ndx[(var,var1)]+var1.dim:con_ndx[(var,var1)]+2*var1.dim]
                    y_sol[(var,var1)] = y1-y2
        else:

            cost = cp.sum(0)
            cons = []
            xs = {}

            for var in self.vars:

                if not var.opt_ready:
                    var.generate_local_QP(self.beta)
                xs[var] = cp.Variable(var.total_dim)
                cost += 0.5*cp.sum_squares(xs[var][var.ndx[var]:var.ndx[var]+var.dim]-var.x_des)
                for i in range(0,var.ns):
                    Aib = var.A[i,:]@ xs[var]-var.b[i]
                    cost += self.beta*cp.log_sum_exp(cp.vstack([ADMM_ETA*Aib,0]))/ADMM_ETA
                cons += [var.Ac@xs[var][var.ndx[var]:var.ndx[var]+var.dim]<=var.bc]

            con_ndx = {}
            for var in self.vars:
                for var1 in var.NB:
                    con_ndx[(var,var1)] = len(cons)
                    cons += [xs[var][var.ndx[var1]:var.ndx[var1]+var1.dim]==xs[var1][var1.ndx[var1]:var1.ndx[var1]+var1.dim]]

            prob = cp.Problem(cp.Minimize(cost),cons)
            prob.solve()


            x_sol = np.empty(0)
            y_sol = {}
            for var in self.vars:
                x_sol = np.append(x_sol,xs[var][var.ndx[var]:var.ndx[var]+var.dim].value)
                for var1 in var.NB:
                    y_sol[(var,var1)] = cons[con_ndx[(var,var1)]].dual_value

        return x_sol,y_sol
