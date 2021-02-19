import numpy as np
import pde_tests as ex
import matplotlib.pyplot as plt
import seaborn as sns 

class Heat_fd:
    def heat1d_fe(self, kappa, ics, bcs, X, T, N, M):
        """
        kappa:  coefficient for u_xx
        f: ics
        X: spatial bounds
        T: end time
        N: no of time intervals
        M: no of spatial intervals 
        """
        a, b = X
        dt = float(T)/N
        dx = (b-a)/M
        mu = dt*kappa/dx**2 

        # u(t,x)
        u = np.zeros([N+1, M+1])
        xs = np.arange(a, b + dx, dx)

        # initial condition u(t=0,x)=f(x)
        u[0] = [ics(x) for x in xs]

        u[:, 0], u[:, -1] = bcs

        l = mu*(dt/dx**2)

        for t in range(N):
            u[t+1, 1:M] = [u[t, n] + l*(u[t, n+1]-2*u[t, n] + u[t, n-1])
                           for n in range(1, M)]   

        print(u[4])
        return u

    def heat1d_fe_matrix(self, kappa, ics, bcs, X, T, N, M):
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

    def heat1d_sam(self, T,IC,BC1,M,N):
        '''
        T - time range [0,T]
        IC - initial condition
        BC1 - BC at x=0
        M - spacial nodes
        N - time nodes
        '''
        delta_t = float(T)/N
        delta_x = 1./M
        spaces  = np.arange(0,1+delta_x,delta_x)
        u = np.zeros((N+1,M+1))
        u[:,0] = BC1 
        u[:,M]= BC1
        u[0,1:M] = IC(spaces[1:M])
        mu = delta_t/delta_x**2
        A = np.diagflat((1-2*mu)*np.ones(M+1))+np.diagflat((mu)*np.ones(M),1)+np.diagflat((mu)*np.ones(M),-1)
        A[0,0]=1
        A[-1,-1]=1
        A[0,1]=0
        A[-1,-2]=0
        for n in range(N):
        # enforcing the boundary condition
        # A single forward Euler timestep
            u[n+1,:] = np.dot(A,u[n,:])
   
        return u
    
    def heat2d_fe_matrix(self, ics, bcs, X, T, N, M):
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

    def heat2d_sam(self, ics, bcs, X, T, N, M):
        """
        ics: ics
        X: spatial bounds (a,b), (c,d)
        T: time bounds (0,T)
        N: no of time intervals
        bcs: (0,0), (0,0)
        M: no of spatial intervals in both dimentions
        """
        (a, b) , (c, d)= X
        F= lambda u , x, y: u[x+1,y]-4*u[x,y]+u[x-1,y] +u[x,y+1]+u[x,y-1]
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
                     
        for t in range(1,M):
            for x in range(1,N):
                for y in range(1,N):
                    u[t,x,y] = u[t-1,x,y] + mu*F(u[t-1],x,y)

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
    u = Heat_fd().heat1d_fe_matrix(
        kappa=1,
        ics=ics,
        bcs=bcs,
        X=X,
        T=T,
        M=M,
        N=N)

    u2 = Heat_fd().heat1d_sam(T=5, IC=ics,BC1=0,M=M,N=N)

def test2():
    ics = lambda x,y : 2*(x+y)
    #bcs = lambda x : np.sin(x) 
    bcs = (0,0), (0,0)
    X = (0, 1), (0, 1)
    N = 5
    M = 5
    T = 7
    u = Heat_fd().heat2d_fe_matrix(
        ics=ics,
        bcs=bcs,
        X=X,
        T=T,
        M=M,
        N=N)

    u1 =  Heat_fd().heat2d_sam(
        ics=ics,
        bcs=bcs,
        X=X,
        T=T,
        M=M,
        N=N)
    print(u)
    print(u)

    



if __name__ == '__main__':
    test2()

   

    



  