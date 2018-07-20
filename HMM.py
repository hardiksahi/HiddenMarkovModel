#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 03:21:51 2018

@author: hardiksahi
"""
import numpy as np

class HMM:
    
    
    #self.c = np.zeros()
    def __init__(self, A,B,pi,obs):
        self.A = A # (N*N)
        self.B = B #(N*M)
        self.pi = pi #(N,)
        self.N = A.shape[0]
        self.M = B.shape[1]
        self.obs = obs
        
        
        
    
    def forwardPass(self):
        timeStamps = self.obs.shape[0] # obs is (T,) object
        
        
        ## Initialization
        c = np.zeros(timeStamps) # (T,)
        alpha = np.zeros((timeStamps,self.A.shape[0])) #(T,N) matrix
        
        
        ## Step1: alpha(t=0)(all states)
        obs0 = self.obs[0]
        B_col_obs0 = self.B[:,obs0] #(N,)
        alphaVector0 = self.pi*B_col_obs0 #(N,)
        
        c_0 = np.sum(alphaVector0)
        #print(c_0)
        c[0] = c_0
        alphaVector0 = alphaVector0/c_0
        alpha[0] = alphaVector0
        #print(alpha, alpha[0].shape)
        #print("c",c)
        
        
        ## Step2: Calculate alpha for timestamps 1 to T-1
        
        for t in range(1,timeStamps):
            
            alphaV = np.dot(np.transpose(self.A), alpha[t-1]) # (N,)
            #print("alphaV", alphaV, alphaV.shape)
            
            obs_t = self.obs[t]
            B_col_obs = self.B[:,obs_t]
            alphaV = alphaV*B_col_obs #(N,)
            #print("alphaV Updated", alphaV, alphaV.shape)
            
            c_t = np.sum(alphaV)
            c[t] = c_t
            alphaV = alphaV/c_t
            
            alpha[t] = alphaV
            
        #print("alpha final", alpha)
        self.c = 1/c
        #print("In forward", self.c)
        self.alpha = alpha
        
        
    def backwardPass(self):
        #print("In back", self.c)
        timeStamps = self.obs.shape[0]
        beta = np.zeros((timeStamps,self.A.shape[0]))
        
        # Step1: For timestamp T-1
        beta[timeStamps-1] = np.full((self.N), self.c[timeStamps-1])
        #print("BETA::", beta)
        
        # Step2: For remaining timesteps:
        
        for t in range(timeStamps-2, -1,-1):
            #print("T is", t)
            obs_t1 = self.obs[t+1]
            B_col_obs = self.B[:,obs_t1] #(N,)
            betaRowT1 = beta[t+1]
            BBeta = B_col_obs*betaRowT1
            #print("Shape:", BBeta.shape) #(N,)
            
            betaVector = np.dot(self.A, BBeta)
            betaVector = betaVector*self.c[t]
            beta[t] = betaVector
            
            #print("Shape of betaVector",betaVector.shape)
        #print("Beta matrix", beta)
        self.beta = beta
        
        
    def forwardBackward(self):
        timeStamps = self.obs.shape[0]
        diGamma = np.zeros((timeStamps-1,self.N,self.N)) #(T-1,N,N)
        gamma = np.zeros((timeStamps, self.N)) #(T,N)
        
        #print("diGamma before", diGamma)
        for t in range(0, timeStamps-1):
            alphaRow = self.alpha[t]
            alphaDiagnol = np.diag(alphaRow)
            
            alphaAMatrix = np.dot(alphaDiagnol, self.A) # (N,N)
            
            obs_t1 = self.obs[t+1]
            B_col_obs = self.B[:,obs_t1] #(N,)
            
            betaT1 = self.beta[t+1]
            BBeta = B_col_obs*betaT1
            
            diGamma[t] = alphaAMatrix*BBeta #(N,N)
            
            gamma[t] = np.sum(alphaAMatrix*BBeta, axis = 1)
        
        
        gamma[timeStamps-1] = self.alpha[timeStamps-1]
        #print("diGamma after", diGamma)
        #print("gamma after",gamma)
        
        self.diGamma = diGamma
        self.gamma = gamma
        
        
    def reEvaluate(self):
        
        # ReEstimate pi..
        self.pi = self.gamma[0]
        #print("A previous", self.A)
        
        # For B we go from 0 to T-1
        denominatorB = np.sum(self.gamma, axis = 0) #(N,)
        
        #print("denominatorB", denominatorB)
        
        
        # Re-estimate A
        # Following is done because for A we go from 0 to T-2
        denominatorA = np.subtract(denominatorB , self.gamma[self.obs.shape[0]-1])
        numA = np.sum(self.diGamma, axis=0)
        #print("numA", numA)
        
        self.A = numA/denominatorA[:, np.newaxis]
        #print("updated A", self.A)
        
        
        # Re-estimate B
        
        for i in range(0, self.N):
            for j in range(0, self.M):
                numB = 0
                #denB = 0
                for t in range(0, self.obs.shape[0]):
                    if self.obs[t] == j:
                        numB = numB+self.gamma[t][i]
                    #denB = denB+self.gamma[t][i]
                
                self.B[i][j] = numB/denominatorB[i]
                #print("den calculated now", denB)
                #print("den cumul", denominatorB[i])
                
        
        #print("Updated B", self.B)
        
        
        
    def calculateLog(self):
        return -np.sum(np.log(self.c))
        
        
                
        
        
        
        
    
    
            
        
        
            
            
        
        
        
        
        
        