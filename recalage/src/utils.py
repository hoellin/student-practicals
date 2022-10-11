#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import Recalage as lib


def results(f,g,nitermax=5000,stepini=1e-2,lam=10,mu=0):
    return lib.RecalageDG(lib.objectif(lib.R(lam,mu),lib.E(f,g)),nitermax,stepini)

def plot_results(f,g,u,CF,step):
    fig, ax = plt.subplots(2,3)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax[0,0].imshow(f)
    ax[0,0].set_title('original function')
    ax[0,1].imshow(g)
    ax[0,1].set_title('target function')
    (ux,uy)=u
    ax[1,0].quiver(ux,uy)
    ax[1,0].set_title('displacement field')
    ax[1,1].imshow(lib.interpol(f,u))
    ax[1,1].set_title('final function')
    ax[0,2].semilogy(CF)
    ax[0,2].set_title('objective history')
    ax[1,2].plot(np.log(step))
    ax[1,2].set_title('step history (log scale)')

    plt.tight_layout()
    plt.show()
    
def obj_vs_lam(f,g,lowerbound=-4, upperbound=2, lam=None, mu=0, nitermax=10000, N_points=20, algo=2, epsi=1e-3):
    if (lam is None):
        param="lambda"
    else:
        param="mu"
    costs = []
    iterations = []
    x_list = []
    for x in np.logspace(lowerbound, upperbound, N_points, endpoint=True):
        if(param=="lambda"):
            lam = x
        else:
            mu = x
        if(algo==2):
            r = lib.R(lam,mu)
            e = lib.E(f,g)
            MC = lib.MoindreCarres(e,r)
            u, CF, step = lib.RecalageGN(MC,nitermax,epsi)
            x_list.append(x)
            costs.append(CF[-1])
            iterations.append(len(CF)-1)
        if(algo==1):
            u, CF, step = results(f,g,nitermax,1e-2,lam,mu)
            x_list.append(x)
            costs.append(CF[-1])
            iterations.append(len(CF)-1)
            
    return (x_list, costs, iterations)

def plot_curves(parameter, x, iterations, costs):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('iterations', color=color)
    ax1.plot(x, iterations, '-o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(x, costs, '-o', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

