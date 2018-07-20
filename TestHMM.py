#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 03:45:28 2018

@author: hardiksahi
"""

import numpy as np
import HMM
np.set_printoptions(threshold=np.nan)

import string
from string import digits
import urllib.request as urllib2
from bs4 import BeautifulSoup
url='http://www.gutenberg.org/cache/epub/13166/pg13166.txt' # Psalm of David
page = urllib2.urlopen(url).read().decode('utf-8-sig')
soup = BeautifulSoup(page,"lxml")
myText=soup.get_text()

remove_digits = str.maketrans('', '', digits)
res = myText.translate(remove_digits)
exclude = set(string.punctuation)
res = ''.join(ch for ch in res if ch not in exclude)

res = " ".join(res.split())

#print(res)

def convertToASCII(alphabet):
    if alphabet == " ":
        return 26
    else:
        return ord(alphabet)-97

obs = np.asarray(list(map(convertToASCII,res.lower())))
#print(np.max(aa))


A = np.array([[0.47468,0.52532],[0.51656,0.48344]])

a1 = np.array([0.03735, 0.03408, 0.03455, 0.03828, 0.03782, 0.03922,0.03688,0.03408,0.03875,0.04062, 0.03735, 0.03968, 0.03548,0.03735, 0.04062,0.03595,0.03641,0.03408,0.04062,0.03548,0.03922,0.04062,0.03455,0.03595,0.03408,0.03408,0.03688])
a2 = np.array([0.03909, 0.03537, 0.03537, 0.03909, 0.03583, 0.03630,0.04048,0.03537,0.03816,0.03909, 0.03490, 0.03723, 0.03537,0.03909, 0.03397,0.03397,0.03816,0.03676,0.04048,0.03443,0.03537,0.03955,0.03816,0.03723,0.03769,0.03955,0.03397])
B = np.array([a1,a2])


N = A.shape[0]
M = B.shape[1]

pi = np.array([0.51316,0.48684])

#obsLength = 4
#obs = np.array([0,1,0,2])

hmm = HMM.HMM(A,B,pi,obs)



iterC = 0
oldLogProb = float('-inf')

while iterC<100:
    print("Iteration", iterC)
    print("Pi")
    print(hmm.pi)
    print("A")
    print(hmm.A)
    print("B")
    print(hmm.B)
    hmm.forwardPass()
    hmm.backwardPass()
    hmm.forwardBackward()
    hmm.reEvaluate()
    
    #print("Gamma")
    #print(hmm.gamma)
    
    logProb = hmm.calculateLog()
    print("logProb", logProb)
    iterC = iterC+1
    
    if logProb<=oldLogProb:
        #print("Final pi", hmm.pi)
        #print("Final A", hmm.A)
        #print("Final B", hmm.B)
        break
    else:
        oldLogProb = logProb
    
    