#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy import interpolate
from PIL import Image
from numpy.linalg import norm as npl_norm

def getGaussians():
	n=21
	sigma=0.3
	[X,Y]=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n), indexing='xy')
	Z=np.sqrt(X*X+Y*Y)
	im1=np.zeros((n,n))
	im1[Z<=.7]=1.
	im1[Z<=.3]=.5
	im1[Z<=.1]=.7
	im2=np.zeros((n,n));
	Z=np.sqrt((X-.3)**2+(Y+.2)**2)
	im2[Z<=.7]=1
	im2[Z<=.3]=.5
	im2[Z<=.1]=.7
	G=np.fft.fftshift(np.exp(-(X**2+Y**2)/sigma**2))
	f=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im1)))
	g=np.real(np.fft.ifft2(np.fft.fft2(G)*np.fft.fft2(im2))) 
	f=f/np.max(f)
	g=g/np.max(g)
	return f,g,(X,Y)

def interpol(f,u):
	# function that computes f \circ Id+u and interpolates it on a mesh
	(ux,uy)=u
	nx,ny=f.shape
	ip=interpolate.RectBivariateSpline(np.arange(nx),np.arange(ny),f)
	[X,Y]=np.meshgrid(np.arange(nx),np.arange(ny), indexing='ij')
	X=X+ux
	Y=Y+uy
	return np.reshape(ip.ev(X.ravel(),Y.ravel()),(nx,ny))

def upscale(f,factor):
	nx,ny=f.shape
	ip=interpolate.RectBivariateSpline(np.arange(nx),np.arange(ny),f)
	[X,Y]=np.meshgrid(np.arange(factor*nx),np.arange(factor*ny), indexing='ij')
	X=X/factor
	Y=Y/factor
	return np.reshape(ip.ev(X.ravel(),Y.ravel()),(factor*nx,factor*ny))

def dx(im):
	d = np.zeros(im.shape)
	d[:-1,:] = im[1:,:]-im[:-1,:]
	return d
def dy(im):
	d = np.zeros(im.shape)
	d[:,:-1] = im[:,1:] - im[:,:-1]
	return d
def dxT(d):
	im = np.zeros(d.shape)
	im[:-1,:] = -d[:-1,:]
	im[1:,:] += d[:-1,:]
	return im
def dyT(d):
	im = np.zeros(d.shape)
	im[:,:-1] = -d[:,:-1]
	im[:,1:] += d[:,:-1]
	return im

class R():
	def __init__(self,lam=10,mu=5):
		self.lam = lam
		self.mu = mu
		self.nb_eval = 0
		self.nb_grad = 0
		self.nb_Hess = 0

	def eval(self,u):
		self.nb_eval+=1
		(ux, uy) = u

		vec1 = (dx(uy)+dy(ux)).ravel()
		vec2 = (dx(ux)+dy(uy)).ravel()

		return 0.5*self.mu*np.dot(vec1,vec1) + 0.5*(self.lam+self.mu)*np.dot(vec2,vec2)

	def grad(self,u):
		self.nb_grad+=1
		(ux,uy) = u

		vec1 = self.mu*(dx(uy)+dy(ux))
		vec2 = (self.lam + self.mu)*(dx(ux)+dy(uy))

		sx = dyT(vec1) + dxT(vec2)
		sy = dxT(vec1) + dyT(vec2)

		return(sx,sy)

	def Hess(self,x):
		assert False
		return None

class E():
	def __init__(self,f,g):
		self.f = f
		self.g = g
		self.nb_eval = 0
		self.nb_grad = 0
		self.nb_Hess = 0

	def eval(self,u):
		self.nb_eval+=1
		return (1/2)*npl_norm(interpol(self.f,u)-self.g)**2

	def grad(self,u):
		self.nb_grad+=1
		scalars = interpol(self.f,u) - self.g
		return (np.multiply(scalars, interpol(dx(self.f),u)), np.multiply(scalars, interpol(dy(self.f),u)))

	def Hess(self,x):
		assert False
		return None

class objectif():
	def __init__(self,r,e):
		self.r=r
		self.e=e

	def eval(self,u):
		return self.e.eval(u) + self.r.eval(u)

	def grad(self,u):
		return tuple(map(sum, zip(self.e.grad(u), self.r.grad(u))))

class MoindreCarres():
	def __init__(self,e,r):
		self.e=e
		self.r=r
		self.obj=objectif(e,r)

	def compute(self,u):
		f = self.e.f
		g = self.e.g
		mu = self.r.mu
		lam = self.r.lam
		(ux, uy) = u

		return (interpol(f,u)-g, np.sqrt(mu)*(dx(uy)+dy(ux)), np.sqrt(mu+lam)*(dx(ux)+dy(uy)))

	def JPsi(self,u,h):
		f = self.e.f
		g = self.e.g
		mu = self.r.mu
		lam = self.r.lam
		(ux, uy) = u
		(hx, hy) = h

		g1 = np.multiply(interpol(dx(f),u), hx) + np.multiply(interpol(dy(f),u), hy)
		g2 = np.sqrt(mu)*(dy(hx)+dx(hy))
		g3 = np.sqrt(mu+lam)*(dx(hx)+dy(hy))

		return (g1,g2,g3)

	def JPsiT(self,u,phi):
		f = self.e.f
		g = self.e.g
		mu = self.r.mu
		lam = self.r.lam
		(ux, uy) = u
		(phi1, phi2, phi3) = phi

		im1 = np.multiply(interpol(dx(f),u), phi1) + np.sqrt(mu)*dyT(phi2) + np.sqrt(mu+lam)*dxT(phi3)
		im2 = np.multiply(interpol(dy(f),u), phi1) + np.sqrt(mu)*dxT(phi2) + np.sqrt(mu+lam)*dyT(phi3)

		return (im1,im2)

	def LM(self,u,h,epsilon=0.):
		(hx, hy) = h
		Jh = self.JPsi(u,h) # (Jh1, Jh2, Jh3)
		(JJ1, JJ2) = self.JPsiT(u,Jh) # (JJ1, JJ2)

		return (epsilon*hx + JJ1, epsilon*hy + JJ2)

def prod_image(im1,im2) :
    return np.dot(im1.ravel(),im2.ravel())
def produit1(u,v) :
    (ux,uy)=u
    (vx,vy)=v
    return prod_image(ux,vx)+prod_image(uy,vy)
def produit2(psi,phi) :
    (psi1,psi2,psi3)=psi
    (phi1,phi2,phi3)=phi
    return prod_image(psi1,phi1)+prod_image(psi2,phi2)+prod_image(psi3,phi3)
		
def linesearch(u,step,descent,cost_old,function) :
    (ux,uy)=u
    (descx,descy)=descent
    step=2*step
    tmp=(ux-step*descx,uy-step*descy)
    cost=function.eval(tmp)
    while cost >cost_old and step > 1.e-8:
        step=0.5*step
        tmp=(ux-step*descx,uy-step*descy)
        cost=function.eval(tmp)
    return tmp,step,cost

def RecalageDG(function,nitermax=500,stepini=0.01) :
    size_image=function.e.f.shape
    u=(np.zeros(size_image),np.zeros(size_image))
    step_list=[]
    niter=0
    step=stepini
    cost=function.eval(u)
    CF=[cost]
    while niter < nitermax and step > 1.e-8 : 
        niter+=1
        grad=function.grad(u)
        u,step,cost=linesearch(u,step,grad,cost,function)
        step_list.append(step)
        CF.append(cost)
        #if (niter % 3 ==0) :
        #    print('iteration {} cost {:1.3e} step {:1.5e}'.format(niter,cost,step))
    return u,np.array(CF),np.array(step_list)

def CGSolve(u,b,epsilon,MC) :
    nitmax=100;
    r=(b[0],b[1])
    d=(np.zeros_like(r[0]),np.zeros_like(r[1]))
    p=(np.copy(r[0]),np.copy(r[1]))
    
    rsold=produit1(r,r)
    for i in range(nitmax) :
        Ap=MC.LM(u,p,epsilon=epsilon)
        alpha=rsold/produit1(r,Ap)
        d=(d[0]+alpha*p[0],d[1]+alpha*p[1])
        r=(r[0]-alpha*Ap[0],r[1]-alpha*Ap[1])
        rsnew=produit1(r,r)
        if np.sqrt(rsnew)<1e-10 :
            return d
        p=(r[0]+rsnew/rsold*p[0],r[1]+rsnew/rsold*p[1])
        rsold=rsnew
    return d
	
def RecalageGN(MC, nitermax, epsilon):
	tol1 = 1e-8
	tol2 = 1e-12
	stepini = 1e-2
	(n,m) = MC.e.f.shape
	u = (np.zeros((n,m)), np.zeros(((n,m))))
	psi = np.array(MC.compute(u))
	grad = MC.JPsiT(u, psi)
	try:
		Hinv = CGSolve(u,grad,epsilon,MC)
	except:
		raise Exception("Singular matrix ; cannot compute H^-1 at step 0")

	step = stepini
	cost = MC.obj.eval(u)

	step_list = [stepini]
	cost_list = [cost]

	output = {}

	niter = 0
	delta = tol2+1
	while niter < nitermax and step > tol1 and delta > tol2:
	#while npl.norm(Hinv) > tol1 and niter < nitermax and delta > tol2:
		u, step, cost = linesearch(u, step, Hinv, cost, MC.obj)
		psi = np.array(MC.compute(u))
		grad = MC.JPsiT(u, psi)
		try:
			Hinv = CGSolve(u,grad,epsilon,MC)
		except:
			raise Exception("Singular matrix ; cannot compute H^-1 at step "+str(niter))
		
		step_list.append(step)
		cost_list.append(cost)
		delta = abs(cost-cost_list[-2])
		
		niter+=1

	return u, np.array(cost_list), np.array(step_list)