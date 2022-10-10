#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for tomographic inversion."""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import fft
from skimage.transform import warp
import math

def printims(ims, nb_ims_per_row = 3, width=11, height=None,
             titles=None, xlabels=None, ylabels=None, extents=None):
    N = len(ims)
    if height is None: height = width*9/12.
    if(N<=nb_ims_per_row):
        f, axarr = plt.subplots(1, N, figsize=(width, height))
        for i in range(N):
            norm_image = cv2.normalize(ims[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            axarr[i].set_title(titles[i]) if titles is not None else None
            axarr[i].set_xlabel(xlabels[i]) if xlabels is not None else None
            axarr[i].set_ylabel(ylabels[i]) if ylabels is not None else None
            if extents is not None:
                if extents[i] is not None:
                    axarr[i].imshow(norm_image, cmap='gray', extent=extents[i], vmin=0, vmax=255)
                else:
                    axarr[i].imshow(norm_image, cmap='gray', vmin=0, vmax=255)
            else:
                axarr[i].imshow(norm_image, cmap='gray', vmin=0, vmax=255)
    else:
        nb_rows = math.ceil(N/nb_ims_per_row)
        f, axarr = plt.subplots(nb_rows, nb_ims_per_row, figsize=(width, height))
        for i in range(nb_rows):
            for j in range(nb_ims_per_row):
                index = i*nb_ims_per_row+j
                norm_image = cv2.normalize(ims[index], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                axarr[i,j].set_title(titles[index]) if titles is not None else None
                axarr[i,j].set_xlabel(xlabels[index]) if xlabels is not None else None
                axarr[i,j].set_ylabel(ylabels[index]) if ylabels is not None else None
                if index<N:
                    if extents is not None:
                        if extents[index] is not None:
                            axarr[i,j].imshow(norm_image, cmap='gray', extent=extents[index], vmin=0, vmax=255)
                    else:
                        axarr[i,j].imshow(norm_image, cmap='gray', vmin=0, vmax=255)
                else:
                    axarr[i,j].imshow(norm_image, cmap='gray', vmin=0, vmax=255)
    f.tight_layout()
    plt.show()

def draw_circle(n):
    circle = np.zeros((n,n))
    thickness = 1+int(n/200.)
    for k in range(n):
        theta = 2*k*np.pi/(n-1)
        i, j = np.clip(int((n/2)*(np.cos(theta)+1)), 0, n-1), np.clip(int((n/2)*(np.sin(theta)+1)), 0, n-1)
        circle[i:i+thickness,j:j+thickness]=255
    return circle

def get_filter(filter_name,length) :
    itv = np.linspace(-length//2,length//2,length)
    fourier_filter = itv*(itv>0) - itv*(itv<0)
    fourier_filter = fourier_filter / (length//2)
        
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fft.fftfreq(length)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, length, endpoint=False)
        cosine_filter = fft.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fft.fftshift(np.hamming(length))
    elif filter_name == "hann":
        fourier_filter *= fft.fftshift(np.hanning(length))
    elif filter_name is None:
        fourier_filter[:] = 0

    return 1-fourier_filter

def rotate_new(im, angle) :
    angle = np.deg2rad(angle)
    
    n,m = np.shape(im)
    center = n // 2
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    T = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                 [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                [0, 0, 1]])
    rotated = warp(im, T, clip=True)
    
    return rotated

