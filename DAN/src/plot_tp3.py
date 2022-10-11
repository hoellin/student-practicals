#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:52:37 2022

@author: thoellin
"""

import sys
import os
import subprocess
#import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import rc

from filters import *
from lin2d_exp import *
from manage_exp import *

x_dim = 2
b_size = 2 # number of experiments to perform
           # below we plot the results for 2 experiments only,
           # but the code still works for b > 2
x0 = get_x0(b_size, 2, sigma0)

T = 200

x = x0
x_list = [x]
current_state = Lin2d(x_dim, 1, None, None)

for t in range(T):
    x = current_state.forward(x)
    x_list.append(x)

x_list_np = np.zeros((T+1, b_size, x_dim))

for i in range(len(x_list)):
    x_list_np[i, :, :] = x_list[i].detach().numpy()

plt.plot(x_list_np[:,0,1], x_list_np[:,0,0])
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.title.set_text('x[0]')
ax1.plot(x_list_np[:,0,0], label="1st simulation")
ax1.plot(x_list_np[:,1,0], label="2nd simulation")
ax1.legend(loc="upper left")
ax2.title.set_text('x[1]')
ax2.plot(x_list_np[:,0,1], label="1st simulation")
ax2.plot(x_list_np[:,1,1], label="2nd simulation")
ax2.legend(loc="upper left")
plt.show()

print("x0 shape: ", x0.detach().numpy().shape)
print("x shape: ", x.detach().numpy().shape)

d = 40
mb = 1
dt = 0.05
F = 8
x0 = torch.ones(d)*F
x0[0] = F + 0.01
x0 = torch.unsqueeze(x0,0)

T = 50

x = x0
x_list = [x]
current_state = EDO(d, mb, dt, "95")

for t in range(T):
    x = current_state.forward(x)
    x_list.append(x)

x_list_np = np.zeros((T+1, 1, d))
for i in range(len(x_list)):
    x_list_np[i, :, :] = x_list[i].detach().numpy()
    
x_list_np = np.reshape(x_list_np, (-1,d))

plt.imshow(x_list_np.T, origin='upper')







