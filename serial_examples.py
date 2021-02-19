import numpy as np
from time import time

class Schemes: 
    soln = []
    
    def __init__(self, v_field, start, stop, h, init_conditions):
        self.vector_field = v_field
        self.ics = init_conditions
        self.a = start
        self.b = stop
        self.times = np.arange(self.a,self.b,h)  
        self.h = h
          
    def prep_preds(scheme):
        def wrapper(self):
            start = time()
            n = self.times.size
            x = np.zeros(n)
            x[0] = self.ics
            self.soln = scheme(self,x)
            end = time()
            return self.times, self.soln, round(end-start, 2)
        return wrapper
 
    @prep_preds
    def euler(self, x):
        for k, t in enumerate(self.times[:-1]):
            x[k+1] = x[k]+self.h*self.vector_field(t,x[k])
        return x
    
    @prep_preds
    def rk4(self, x):
        for k, t in enumerate(self.times[:-1]):
            f1 = self.vector_field(t,x[k])
            f2 = self.vector_field(t+0.5*self.h, x[k]+0.5*self.h*f1)
            f3 = self.vector_field(t+0.5*self.h, x[k]+0.5*self.h*f2)
            f4 = self.vector_field(t+self.h, x[k]+self.h*f3)
            x[k+1] = x[k]+self.h/6*(f1+2*f2+2*f3+f4)
        return x

    @prep_preds
    def ab4(self,x):
        for k,t in enumerate(self.times[:-1]):
            if (k < 3):
                f1 = self.vector_field(t,x[k])
                f2 = self.vector_field(t+0.5*self.h, x[k]+0.5*self.h*f1)
                f3 = self.vector_field(t+0.5*self.h, x[k]+0.5*self.h*f2)
                f4 = self.vector_field(t+self.h, x[k]+self.h*f3)
                x[k+1] = x[k]+self.h/6*(f1+2*f2+2*f3+f4)
                #x[k+1] = x[k]+self.h*self.vector_field(t,x[k])
            else:    
                f1 = self.vector_field(t,x[k])
                f2 = self.vector_field(self.times[k-1], x[k-1])
                f3 = self.vector_field(self.times[k-2], x[k-2])
                f4 = self.vector_field(self.times[k-3], x[k-3])
                x[k+1] = x[k] + self.h/24 * (55*f1 - 59*f2 + 37*f3 - 9*f4)
        return x
