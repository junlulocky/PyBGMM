

import numpy as np
import random
from matplotlib import pyplot as plt

class ARS():
    '''
    This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
    Where possible, naming convention has been borrowed from this paper.
    The PDF must be log-concave.

    Currently does not exploit lower hull described in paper- which is fine for drawing
    only small amount of samples at a time.
    '''
    
    def __init__(self, f, fprima, xi=[0.1,1,40], lb=-np.Inf, ub=np.Inf, use_lower=False, ns=50, **fargs):
        '''
        initialize the upper (and if needed lower) hulls with the specified params
        
        Parameters
        ==========
        f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
           density we want to sample from
        fprima:  d/du log(f(u,...))
        xi: ordered vector of starting points in wich log(f(u,...) is defined
            to initialize the hulls
        D: domain limits
        use_lower: True means the lower sqeezing will be used; which is more efficient
                   for drawing large numbers of samples
        
        
        lb: lower bound of the domain
        ub: upper bound of the domain
        ns: maximum number of points defining the hulls
        fargs: arguments for f and fprima
        '''
        
        self.lb = lb
        self.ub = ub
        self.f = f
        self.fprima = fprima
        self.fargs = fargs
        
        #set limit on how many points to maintain on hull
        self.ns = 50
        self.x = np.array(xi) # initialize x, the vector of absicassae at which the function h has been evaluated
        self.h = self.f(self.x, **self.fargs)
        self.hprime = self.fprima(self.x, **self.fargs)
        self.no = False
        self.nosample = False

        #Avoid under/overflow errors. the envelope and pdf are only
        # proporitional to the true pdf, so can choose any constant of proportionality.
        self.offset = np.amax(self.h)
        self.h = self.h-self.offset 

        # Derivative at first point in xi must be > 0
        # Derivative at last point in xi must be < 0
        if not(self.hprime[0] > 0):
            print self.hprime
            self.no = True
            #raise IOError('initial anchor points must span mode of PDF')
        if not(self.hprime[-1] < 0):
            self.x = self.x + 0.01
            self.no = True
            #raise IOError('initial anchor points must span mode of PDF')
        self.insert() 

        
    def draw(self, N):
        '''
        Draw N samples and update upper and lower hulls accordingly
        '''
        if self.no:
            return 1
        samples = np.zeros(N)
        n=0
        while n < N:
            [xt,i] = self.sampleUpper()
            # TODO (John): Should perform squeezing test here but not yet implemented 
            ht = self.f(xt, **self.fargs)
            hprimet = self.fprima(xt, **self.fargs)
            ht = ht - self.offset
            #ut = np.amin(self.hprime*(xt-x) + self.h);
            ut = self.h[i] + (xt-self.x[i])*self.hprime[i]

            # Accept sample? - Currently don't use lower
            u = random.random()
            if u < np.exp(ht-ut) or self.nosample:
                samples[n] = xt
                n +=1

            # Update hull with new function evaluations
            if self.u.__len__() < self.ns:
                self.insert([xt],[ht],[hprimet])
            
        return samples

    
    def insert(self,xnew=[],hnew=[],hprimenew=[]):
        '''
        Update hulls with new point(s) if none given, just recalculate hull from existing x,h,hprime
        '''
        if xnew.__len__() > 0:
            x = np.hstack([self.x,xnew])
            idx = np.argsort(x)
            self.x = x[idx]
            self.h = np.hstack([self.h, hnew])[idx]
            self.hprime = np.hstack([self.hprime, hprimenew])[idx]

        self.z = np.zeros(self.x.__len__()+1)
        
        # This is the formula explicitly stated in Gilks. 
        # Requires 7(N-1) computations 
        # Following line requires 6(N-1)
        # self.z[1:-1] = (np.diff(self.h) + self.x[:-1]*self.hprime[:-1] - self.x[1:]*self.hprime[1:]) / -np.diff(self.hprime); 

        self.z[1:-1] = (np.diff(self.h) - np.diff(self.x*self.hprime))/-np.diff(self.hprime) 
        
        self.z[0] = self.lb; self.z[-1] = self.ub
        N = self.h.__len__()
        self.u = self.hprime[[0]+range(N)]*(self.z-self.x[[0]+range(N)]) + self.h[[0]+range(N)]

        self.s = np.hstack([0,np.cumsum(np.diff(np.exp(self.u))/self.hprime)])
        self.cu = self.s[-1]


    def sampleUpper(self):
        '''
        Return a single value randomly sampled from the upper hull and index of segment
        '''
        u = random.random()
        
        # Find the largest z such that sc(z) < u
        try:
            i = np.nonzero(self.s / self.cu < u)[0][-1]
        except:
            i = 1
            self.nosample = True
            print 'no sample upper index'
        #i = np.nonzero(self.s/self.cu < u)[0][-1]

        # Figure out x from inverse cdf in relevant sector
        xt = self.x[i] + (-self.h[i] + np.log(self.hprime[i]*(self.cu*u - self.s[i]) + 
        np.exp(self.u[i]))) / self.hprime[i]

        return [xt,i]

    def plotHull(self):
        '''
        Plot the piecewise linear hull using matplotlib
        '''
        xpoints = self.z
        #ypoints = np.hstack([0,np.diff(self.z)*self.hprime])
        ypoints = np.exp(self.u) 
        plt.plot(xpoints,ypoints)
        plt.show()
        '''
        for i in range(1,self.z.__len__()):
            x1 = self.z[i]
            y1 = 0
            x2 = self.z[i+1]
            y2 = self.z[i+1]-self.z[i] * hprime[i]
        '''
