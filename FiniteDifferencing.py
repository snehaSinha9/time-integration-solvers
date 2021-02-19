import numpy as np
import pde_tests as ex
import matplotlib.pyplot as plt

class Heat_fd:

    def heat1d_fe(self, kappa, ics, bcs, X, T, N, M):
        """
        kappa:  coefficient for u_xx
        ics: ics
        bcs: bcs start and end
        X: spatial bounds
        T: time bounds
        N: no of time intervals
        M: no of spatial intervals 
        """
        a, b = X
        dt = float(T)/N
        dx = (b-a)/M
        mu = dt*kappa/dx**2 
        xs = np.arange(a, b + dx, dx)

        # u(t,x)
        u = np.zeros([N+1,M+1])
        u[:,0] = bcs[0] 
        u[:,M]= bcs[1]
        u[0] = list(map(ics, xs))
        
        A = np.diagflat([mu]*(M-2), -1) + np.diagflat([1-2*mu]*(M-1)) \
            + np.diagflat([mu]*(M-2), 1) 
        for n in range(N):
            u[n+1,1:M] = np.dot(A, u[n,1:M])

        return  u 

    def heat2d_fe(self, ics, bcs, X, T, N, M):
        """
        ics: ics
        X: spatial bounds (a,b), (c,d)
        T: time bounds (0,T)
        N: no of time intervals
        bcs: (0,0), (0,0)
        M: no of spatial intervals in both dimentions
        """
        (a, b) , (c, d)= X

        dt = float(T)/N
        dx = (b-a)/M
        dy = (d-c)/M
        mu = dt/dx**2 
        xs = np.arange(a, b + dx, dx)
        ys = np.arange(c, d+dy, dy)

        # u(t,x) bcs
        u = np.zeros([N+1, M+1, M+1])
        u[:,0], u[:,:,0] = bcs[0][0], bcs[1][0]
        u[:,M], u[:,:,M] = bcs[0][1], bcs[1][1]

        # ics 
        for i in range(1,M):
            for j in range(1,M):
                u[0,i,j] = ics(xs[i], ys[j])
                
        
        A = np.diagflat([mu]*(M-2), -1) + np.diagflat([1-2*mu]*(M-1)) \
            + np.diagflat([mu]*(M-2), 1) 

        for n in range(N):
            u[n+1,1:M] = np.dot(A, u[n,1:M])

        return  u 

    def heat1d_ridc(self):
        pass


def test1():
     # neuman (only when du_dx=0)
    def ics(x): return x*(1-x)
    bcs = (0, 0)
    X = (0, 1)
    N = 10
    M = 10
    T = 5
    u = Heat_fd().heat1d_fe(
        kappa=1,
        ics=ics,
        bcs=bcs,
        X=X,
        T=T,
        M=M,
        N=N)


def test2():
    ics = lambda x,y : 2*(x+y)
    #bcs = lambda x : np.sin(x) 
    bcs = (0,0), (0,0)
    X = (0, 1), (0, 1)
    N = 5
    M = 5
    T = 7
    u = Heat_fd().heat2d_fe(
        ics=ics,
        bcs=bcs,
        X=X,
        T=T,
        M=M,
        N=N)
    print(u)

if __name__ == '__main__':
    test2()

   

    



  