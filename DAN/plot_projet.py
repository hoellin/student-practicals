#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:49:08 2022

@author: hoellinger
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pkl

plt.rcParams['text.usetex'] = True

res_dir = "lin2d_exp/"

net = torch.load(res_dir+'net.pt')
test_sample_x = pkl.load(open(res_dir+'test_sample_x.pkl', 'rb'))
test_sample_y = pkl.load(open(res_dir+'test_sample_y.pkl', 'rb'))
analysis = pkl.load(open(res_dir+'analysis.pkl', 'rb'))


analysis = torch.stack([xa_t.mean for xa_t in analysis]).detach().cpu().numpy()

mu_a = np.mean(analysis, axis=1)

plt.figure(figsize=(12, 8))
plt.title(r"Dynamics and estimated trajectory in the 2D periodic hamiltonian system", fontsize=20)
plt.plot(test_sample_x[:, 0], test_sample_x[:, 1], c='r', marker='.', label=r"$x_t$ (True)")
plt.scatter(test_sample_y[:, 0], test_sample_y[:, 1], c='g', marker='x', label=r"$y_t$ (Observations)")
plt.scatter(mu_a[:, 0], mu_a[:, 1], c='b', marker='o', label=r"$\mu_t^\mathbf{a}$ (Analyse)")
plt.legend()
plt.show()

print("Loss finale = ", np.mean(np.array(net.scores["LOSS"])[-50:])) # take -100:-50 if generalisation test