import pdb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
from scipy.io import loadmat
from scipy import interpolate, sparse
import math
from numpy.linalg import norm
from scipy.linalg import block_diag
from cvxopt import matrix, solvers
from random import sample
from itertools import combinations
import cvxpy as cp
from scipy.io import savemat
from LCADMM import ADMM_var,ADMM_problem
from LCADMM.constants import *



def main_basic():

    N_trials = 5
    maxiter = 5000
    y_err_rec = np.zeros([N_trials,maxiter])
    x_err_rec = np.zeros([N_trials,maxiter])
    dx_rec = np.zeros([N_trials,maxiter])

    for k in range(0,N_trials):
        vars = []
        lb = -np.ones(2)*100
        ub = np.ones(2)*100
        m = 2
        N = 8
        M = 10
        ndx = {}
        dim = 0
        x_des = 0.5*np.ones(2)
        prob = ADMM_problem()
        for i in range(0,N):
            vars.append(ADMM_var(2,lb = lb, ub = ub, x_des=x_des))
            ndx[vars[-1]] = dim
            dim+=vars[-1].dim
        for i in range(0,M):
            pair = sample(vars,2)
            A0 = np.random.randn(m,pair[0].dim)
            A1 = np.random.randn(m,pair[1].dim)
            b = np.random.randn(m)
            prob.Add_ineq_cons(pair,[A0,A1],b)


        for i in range(0,3):
            triple = sample(vars,3)
            A0 = np.random.randn(m,triple[0].dim)
            A1 = np.random.randn(m,triple[1].dim)
            A2 = np.random.randn(m,triple[2].dim)
            b = np.random.randn(m)
            prob.Add_ineq_cons(triple,[A0,A1,A2],b)




        # for var in vars:
        #     prob.add_agent(var)
        # prob.solve()

        x_solc,y_solc = prob.solve_constrained_centralized()
        x_solc1 = prob.solve_centralized()
        for var in vars:
            var.generate_local_QP(prob.beta)



        ##
        iter = 0
        err = 1.

        tol = 1e-4

        for var in prob.vars:
            var.solve_local_QP(prob.beta)
        while iter<maxiter and err>tol:
            y_err = 0.
            x_err = 0.
            iter +=1
            err = 0
            for var in prob.vars:
                # var.solve_local_QP()
                delta_x = var.x_sol[0:var.total_dim]-var.x_sol_old
                hh = 0
                hh +=(var.p+ADMM_RHO*var.N_nb)*norm(delta_x)**2
                du = hh
                for var1 in var.NB:

                    dx = (var.x_sol[var.ndx[var1]:var.ndx[var1]+var1.dim]-var1.x_sol[var1.ndx[var1]:var1.ndx[var1]+var1.dim]).flatten()
                    hh+=norm(dx)**2*(2-ADMM_GAMMA)*ADMM_RHO+2*ADMM_RHO*dx.dot(delta_x[var.ndx[var1]:var.ndx[var1]+var1.dim])
                    du +=ADMM_RHO*ADMM_GAMMA*norm(dx)**2
                    var.y1[var1]+=ADMM_GAMMA*ADMM_RHO*dx/var1.x_norm
                    var1.y2[var] = var.y1[var1]
                    var1.z[var] = var.x_sol[var.ndx[var]:var.ndx[var]+var.dim].flatten()
                    err = max(err,norm(dx/var1.x_norm))


                    y_err += norm(y_solc[(var,var1)]-var.y1[var1])
                x_err+=norm((var.x_sol[var.ndx[var]:var.ndx[var]+var.dim]-x_solc[ndx[var]:ndx[var]+var.dim])/var.x_norm)

            y_err_rec[k][iter-1] = (y_err/M)
            x_err_rec[k][iter-1] = (x_err/N)
            dx_rec[k][iter-1] = err

            print(err)
            for var in prob.vars:
                var.solve_local_QP(prob.beta)

        prob.x_sol = np.zeros(prob.vardim)
        for var in prob.vars:
            prob.x_sol[prob.ndx[var]:prob.ndx[var]+var.dim]= var.x_sol[var.ndx[var]:var.ndx[var]+var.dim].flatten()
        y_err_rec[k] = y_err_rec[k]/y_err_rec[k][0]
        x_err_rec[k] = x_err_rec[k]/x_err_rec[k][0]
        dx_rec[k] = dx_rec[k]/dx_rec[k][0]

    pdb.set_trace()
    # savevars = {'x_err_rec':x_err_rec,'y_err_rec':y_err_rec,'dx_rec':dx_rec}
    # savemat("res.mat", savevars)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for k in range(0,N_trials):
        ax1.plot(list(range(0,len(x_err_rec[k]))),x_err_rec[k], 'g--', linewidth=1)
        ax2.plot(list(range(0,len(y_err_rec[k]))),y_err_rec[k], 'g--', linewidth=1)
    plt.show()


    pdb.set_trace()


def main_perturbation():
    vars = []
    lb = -np.ones(2)
    ub = np.ones(2)
    m = 4
    N = 8
    M = 15
    ndx = {}
    dim = 0
    x_des = 0.5*np.ones(2)
    Ain0 = {}
    bin0 = {}
    Ain1 = {}
    bin1 = {}

    for i in range(0,N):
        vars.append(ADMM_var(2,lb = lb, ub = ub, x_des=x_des))
        ndx[vars[-1]] = dim
        dim+=vars[-1].dim

    for i in range(0,M):
        pair = sample(vars,2)
        A0 = np.random.randn(m,pair[0].dim)
        A1 = np.random.randn(m,pair[1].dim)
        b = np.random.randn(m)
        Ain0[tuple(pair)] = (A0,A1)
        bin0[tuple(pair)] = b

        A0 = np.random.randn(m,pair[0].dim)
        A1 = np.random.randn(m,pair[1].dim)
        b = np.random.randn(m)

        Ain1[tuple(pair)] = (A0,A1)
        bin1[tuple(pair)] = b

    ss = list(np.linspace(0,1,101))

    var00 = pair[0]
    var11 = pair[1]
    x0_sol = np.empty([0,var00.dim])
    y01_sol = np.empty([0,var11.dim])
    x0_sol_relax = np.empty([0,var00.dim])
    y01_sol_relax = np.empty([0,var11.dim])

    for s in ss:
        for var in vars:
            var.reset(reset_dual = True)
        for pair in Ain0.keys():
            A0 = s*Ain0[pair][0]+(1-s)*Ain1[pair][0]
            A1 = s*Ain0[pair][1]+(1-s)*Ain1[pair][1]
            b = s*bin0[pair]+(1-s)*bin1[pair]
            pair[0].Add_ineq_cons(pair[1],A0,A1,b)
            pair[1].Add_ineq_cons(pair[0],A1,A0,b)




        prob = ADMM_problem()
        for var in vars:
            prob.add_agent(var)
        x_solc,y_solc = prob.solve_constrained_centralized(exp_relax=False)
        x0_sol = np.vstack((x0_sol,x_solc[ndx[var00]:ndx[var00]+var00.dim]))
        y01_sol = np.vstack((y01_sol,y_solc[(var00,var11)]))
        x_solc,y_solc = prob.solve_constrained_centralized(exp_relax=True)
        x0_sol_relax = np.vstack((x0_sol_relax,x_solc[ndx[var00]:ndx[var00]+var00.dim]))
        y01_sol_relax = np.vstack((y01_sol_relax,y_solc[(var00,var11)]))
        print(s)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(np.array(ss),x0_sol[:,0], 'g--', linewidth=1)
    ax1.plot(np.array(ss),x0_sol[:,1], 'g--', linewidth=1)
    ax1.plot(np.array(ss),x0_sol_relax[:,0], 'r--', linewidth=1)
    ax1.plot(np.array(ss),x0_sol_relax[:,1], 'r--', linewidth=1)
    ax2.plot(np.array(ss),y01_sol[:,0], 'g--', linewidth=1)
    ax2.plot(np.array(ss),y01_sol[:,1], 'g--', linewidth=1)
    ax2.plot(np.array(ss),y01_sol_relax[:,0], 'r--', linewidth=1)
    ax2.plot(np.array(ss),y01_sol_relax[:,1], 'r--', linewidth=1)

    plt.show()
    savevars = {'x0_sol':x0_sol,'x0_sol_relax':x0_sol_relax,'y01_sol':y01_sol,'y01_sol_relax':y01_sol_relax}
    savemat("smoothing.mat", savevars)
    pdb.set_trace()


def main_perturbation_decentralized():
    vars = []
    lb = -np.ones(2)
    ub = np.ones(2)
    m = 4
    N = 8
    M = 15
    ndx = {}
    dim = 0
    x_des = 0.5*np.ones(2)
    Ain0 = {}
    bin0 = {}
    Ain1 = {}
    bin1 = {}

    for i in range(0,N):
        vars.append(ADMM_var(2,lb = lb, ub = ub, x_des=x_des))
        ndx[vars[-1]] = dim
        dim+=vars[-1].dim

    for i in range(0,M):
        pair = sample(vars,2)
        A0 = np.random.randn(m,pair[0].dim)
        A1 = np.random.randn(m,pair[1].dim)
        b = np.random.randn(m)
        Ain0[tuple(pair)] = (A0,A1)
        bin0[tuple(pair)] = b

        A0 = np.random.randn(m,pair[0].dim)
        A1 = np.random.randn(m,pair[1].dim)
        b = np.random.randn(m)

        Ain1[tuple(pair)] = (A0,A1)
        bin1[tuple(pair)] = b

    ss = list(np.linspace(0,1,51))

    var00 = pair[0]
    var11 = pair[1]
    x0_sol = np.empty([0,var00.dim])
    y01_sol = np.empty([0,var11.dim])
    x0_sol_relax = np.empty([0,var00.dim])
    y01_sol_relax = np.empty([0,var11.dim])
    x_err_rec = np.empty(0)
    y_err_rec = np.empty(0)

    for s in ss:
        for var in vars:
            var.reset(reset_dual = False)
        for pair in Ain0.keys():
            A0 = s*Ain0[pair][0]+(1-s)*Ain1[pair][0]
            A1 = s*Ain0[pair][1]+(1-s)*Ain1[pair][1]
            b = s*bin0[pair]+(1-s)*bin1[pair]
            pair[0].Add_ineq_cons(pair[1],A0,A1,b)
            pair[1].Add_ineq_cons(pair[0],A1,A0,b)




        prob = ADMM_problem()
        for var in vars:
            prob.add_agent(var)
        x_sold = prob.solve(maxiter = 100)
        x_solc,y_solc = prob.solve_constrained_centralized(exp_relax=False)
        x0_sol = np.vstack((x0_sol,x_solc[ndx[var00]:ndx[var00]+var00.dim]))
        y01_sol = np.vstack((y01_sol,y_solc[(var00,var11)]))
        x_err = 0
        y_err = 0
        x_err = norm(x_sold-x_solc)/N

        for var in vars:
            for var1 in var.NB:
                y_err += norm(var.y1[var1]-y_solc[(var,var1)])
        y_err = y_err/M
        x_err_rec = np.append(x_err_rec,x_err)
        y_err_rec = np.append(y_err_rec,y_err)





        # x_solc,y_solc = prob.solve_constrained_centralized(exp_relax=True)
        # x0_sol_relax = np.vstack((x0_sol_relax,x_solc[ndx[var00]:ndx[var00]+var00.dim]))
        # y01_sol_relax = np.vstack((y01_sol_relax,y_solc[(var00,var11)]))
        print(s)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(np.array(ss),x0_sol[:,0], 'g--', linewidth=1)
    # ax1.plot(np.array(ss),x0_sol[:,1], 'g--', linewidth=1)
    ax1.plot(np.array(ss),x_err_rec, 'r--', linewidth=1)
    ax2.plot(np.array(ss),y01_sol[:,0], 'g--', linewidth=1)
    # ax2.plot(np.array(ss),y01_sol[:,1], 'g--', linewidth=1)
    ax2.plot(np.array(ss),y_err_rec, 'r--', linewidth=1)


    plt.show()




    # x_sold = prob.solve()

    # x_solc1 = prob.solve_centralized()





    pdb.set_trace()


if __name__ == '__main__':
    main_perturbation_decentralized()
    # main_perturbation()
    # main_basic()
