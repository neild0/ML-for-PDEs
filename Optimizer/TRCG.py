# The implementation of the Trust-Region (TR) Newton CG method.

import os
import time
import datetime
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1) 
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
import sys


class TRCGOptimizer:
    def __init__(self,model,device,radius,cgopttol=1e-7,c0tr=0.2,c1tr=0.25,c2tr=0.75,t1tr=0.25,t2tr=2.0,radius_max=5.0,\
                 radius_initial=0.1):
        
        self.model = model
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = self.model.numberOfParameters()
        self.cgmaxiter = 60
        
    def findroot(self,x,p):
        
        aa = 0.0
        bb = 0.0
        cc = 0.0
    
        for e in xrange(len(x)):
            aa += (p[e]*p[e]).sum()
            bb += (p[e]*x[e]).sum()
            cc += (x[e]*x[e]).sum()
        
        bb = bb*2.0
        cc = cc - self.radius**2
    
        alpha = (-2.0*cc)/(bb+(bb**2-(4.0*aa*cc)).sqrt())

        return alpha.data.item()
    
    def computeListNorm(self,lst):
        return np.sum([(ri*ri).sum() for ri in lst])**0.5
    
    def computeListNormSq(self,lst):
        return  np.sum([ (ri*ri).sum() for ri in lst]) 
    
    def CGSolver(self,loss_grad,cnt_compute):
    
        cg_iter = 0 # iteration counter
        x0 = [i.data*0 for i in self.model.parameters()]
        r0 = [i.data+0.0 for i in loss_grad]  # set initial residual to gradient
        p0 = [-i.data+0.0 for i in loss_grad] # set initial conjugate direction to -r0
        
        self.cgopttol = self.computeListNormSq(loss_grad)
        self.cgopttol = self.cgopttol.data.item()**0.5
        self.cgopttol = (min(0.5,self.cgopttol**0.5))*self.cgopttol
    
        cg_term = 0
        j = 0

        while 1:
            j+=1
    
            # if CG does not solve model within max allowable iterations
            if j > self.cgmaxiter:
                j=j-1
                p1 = x0
                print '\n\nCG exceeds the iteration limit !!!\n\n'
                break
                
            # hessian vector product
            cnt_compute+=1
            
            loss_grad_direct = np.sum([(gi*si).sum() for gi, si in zip(loss_grad,p0)])
            Hp = torch.autograd.grad(loss_grad_direct,self.model.parameters(),retain_graph=True) # hessian-vector in tuple
            pHp = np.sum([(Hpi*p0i).sum() for Hpi, p0i in zip(Hp,p0)])
    
            # if nonpositive curvature detected, go for the boundary of trust region
            if pHp.data.item() <= 0:
                tau = self.findroot(x0,p0)
                p1 = [xi+tau*p0i  for xi, p0i in zip(x0,p0)]
                cg_term = 1
                break
            
            # if positive curvature
            # vector product
            rr0 = self.computeListNormSq(r0)
            
            # update alpha
            alpha = (rr0/pHp).data.item()
        
            x1 = [xi+alpha*pi for xi,pi in zip(x0,p0)]
            norm_x1 = self.computeListNorm(x1)
            
            # if norm of the updated x1 > radius
            if norm_x1.data.item() >= self.radius:
                tau = self.findroot(x0,p0)
                p1 = [xi+tau*pi for xi,pi in zip(x0,p0)]
                cg_term = 2
                break
    
            # update residual
            r1 = [ri+alpha*Hpi for ri, Hpi in zip(r0, Hp)]
            norm_r1 = self.computeListNorm(r1)
    
            if norm_r1.data.item() < self.cgopttol:
                p1 = x1
                cg_term = 3
                break
    
            rr1 = self.computeListNormSq(r1)
            beta = (rr1/rr0).data.item()
    
            # update conjugate direction for next iterate
            p1 = [-ri+beta*pi for ri,pi in zip(r1,p0)]
    
            p0 = p1
            x0 = x1
            r0 = r1
    

        cg_iter = j
        d = p1

        return d,cg_iter,cg_term,cnt_compute
    
    def step(self,loss,MSE,x_time_series,y_time_series,resblock):
        
        update = 2
    
        loss_grad = torch.autograd.grad(loss,self.model.parameters(),create_graph=True) 
    
        cnt_compute=1
        
        # Conjugate Gradient Method
        d, cg_iter, cg_term, cnt_compute = self.CGSolver(loss_grad,cnt_compute)

        # current iterate 
        w0 = [m.data+0.0 for m in self.model.parameters()]
    
        # update solution
        for m,di in zip(self.model.parameters(),d):
            m.data.set_(m.data+0.0+di)
    
        # MSE loss
        with torch.no_grad():
            Pred_new = self.model(x_time_series,resblock)
            loss_new = MSE(Pred_new,y_time_series)

        numerator = loss.data.item() - loss_new.data.item()

        # dHd   
        loss_grad_direct = np.sum([(gi*di).sum() for gi, di in zip(loss_grad,d)])
        Hd = torch.autograd.grad(loss_grad_direct,self.model.parameters()) # hessian-vector in tuple
        dHd = np.sum([(Hdi*di).sum() for Hdi, di in zip(Hd,d)])
        
        gd = np.sum([(gi*di).sum() for gi, di in zip(loss_grad,d)])

        norm_d = self.computeListNorm(d)
        
        denominator = -gd.data.item() - 0.5*(dHd.data.item())

        # ratio
        rho = numerator/denominator

        if rho < self.c1tr: # shrink radius
            self.radius = self.t1tr*self.radius
            update = 0
        if rho > self.c2tr and np.abs(norm_d.data.item() - self.radius) < 1e-10: # enlarge radius
            self.radius = min(self.t2tr*self.radius,self.radius_max)
            update = 1
        # otherwise, radius remains the same
        
        if rho <= self.c0tr: # reject d
            update = 3
            ind = 0
            for wi,mi in zip(w0,self.model.parameters()):
                mi.data.set_(wi.data)
                
        return self.radius, cnt_compute, cg_iter
        