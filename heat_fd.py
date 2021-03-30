import numpy as np
import matplotlib.pyplot as plt
from dc_experiments import DCs as dc
from scipy.interpolate import lagrange
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


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
        u = np.zeros([N+1, M+1])
        u[:, 0] = bcs[0]
        u[:, M] = bcs[1]
        u[0] = list(map(ics, xs))

        A = np.diagflat([mu]*(M-2), -1) + np.diagflat([1-2*mu]*(M-1)) \
            + np.diagflat([mu]*(M-2), 1)
        for n in range(N):
            u[n+1, 1:M] = np.dot(A, u[n, 1:M])

        return u

    def heat2d_fe(self, ics, bcs, X, T, N, M):
        """
        ics: ics
        X: spatial bounds (a,b), (c,d)
        T: time bounds (0,T)
        N: no of time intervals
        bcs: (0,0), (0,0)
        M: no of spatial intervals in both dimentions
        """
        (a, b), (c, d) = X

        dt = float(T)/N
        dx = (b-a)/M
        dy = (d-c)/M
        mu = dt/dx**2
        xs = np.arange(a, b + dx, dx)
        ys = np.arange(c, d+dy, dy)

        # u(t,x) bcs
        u = np.zeros([N+1, M+1, M+1])
        u[:, 0], u[:, :, 0] = bcs[0][0], bcs[1][0]
        u[:, M], u[:, :, M] = bcs[0][1], bcs[1][1]

        # ics
        for i in range(1, M):
            for j in range(1, M):
                u[0, i, j] = ics(xs[i], ys[j])

        A = np.diagflat([mu]*(M-2), -1) + np.diagflat([1-2*mu]*(M-1)) \
            + np.diagflat([mu]*(M-2), 1)

        for n in range(N):
            u[n+1, 1:M] = np.dot(A, u[n, 1:M])

        return u

    def heat1d_ridc(self, kappa, ics, T, N, M, L, bcs=(0, 0), X=(0., 1.)):
        """
        kappa: coefficient for heat equation
        ics: ics, y0
        X: spatial bounds (a=0,b=1)
        T: time bounds (0,T)
        N: no. of time intervals
        bcs: (0,0)
        M: no. of spatial nodes
        L: no. of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        """

        a, b = X
        dt = float(T)/N
        dx = (b-a)/M
        mu = dt*kappa/dx**2
        xs = np.arange(a, b + dx, dx)
        t = np.arange(0, T+dt, dt)

        A = np.diagflat([mu]*M, -1) + np.diagflat([1-2*mu]
                                                  * (M+1)) + np.diagflat([mu]*M, 1)
        A[0, 0], A[-1, -1], A[0, 1], A[-1, -2] = 1, 1, 0, 0

        # forward Euler discretisation
        def func(u_): return [u_[0]] + [(u_[i-1]-2*u_[i] +
                                         u_[i+1])/(dx**2) for i in range(1, M)] + [u_[-1]]

        # Note Lm is the number of correctors
        Lm = L - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Lm, Lm+1])
        for m in range(Lm):  # Calculate qudrature weights
            for i in range(Lm+1):
                x = np.arange(Lm+1)  # Construct a polynomial
                # which equals to 1 at i, 0 at other points
                y = np.zeros(Lm+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Lm+2)
                for k in range(Lm+1):
                    P[k] = para[k]/(Lm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Lm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, M+1])
        # ics and bcs
        yy[0, 1:M] = ics(xs[1:M].copy())
        yy[0, 0], yy[0, M] = bcs

        # Value of RHS at initial time
        F0 = func(yy[0])
        F1 = np.zeros([Lm, L, M+1])
        F1[:, 0] = np.array([F0]*Lm)

        F2 = F0
        # Y2 [L] new point derived in each level (prediction and corrections)
        Y2 = np.array([yy[0]]*L)

        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to L points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, L-1):
            # KK1 = F1[0, iTime]
            # KK2 = func(t[iTime]+dt/2, Y2[0]+KK1*dt/2)
            # KK3 = func(t[iTime]+dt/2, Y2[0]+KK2*dt/2)
            # KK4 = func(t[iTime]+dt,   Y2[0]+KK3*dt)
            # Y2[0] = Y2[0] + dt*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
            # F1[0, iTime+1] = func(t[iTime+1], Y2[0])
            Y2[0] = np.dot(A, Y2[0])
            F1[0, iTime+1] = func(Y2[0])

        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, L-1):
            ll = iCor - 1
            for iTime in range(0, L-1):
                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*(F1[iCor, iTime, 1:M]-F1[ll, iTime, 1:M]) + \
                    dt * np.dot(S[iTime], F1[ll, :, 1:M])
                F1[iCor, iTime+1] = func(Y2[iCor])

        # treat the last correction loop a little different
        for iTime in range(0, L-1):
            Y2[L-1, 1:M] = Y2[L-1, 1:M] + dt*(F2[1:M]-F1[L-2, iTime, 1:M]) + \
                dt * np.dot(S[iTime], F1[L-2, :, 1:M])
            F2 = func(Y2[L-1])
            yy[iTime+1] = Y2[L-1]

        # ================== INITIAL PART (2) ==================

        for iTime in range(L-1, 2*L-2):
            iStep = iTime - (L-1)
            # prediction loop
            Y2[0] = np.dot(A, Y2[0])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*(F1[iCor, -1, 1:M]-F1[ll, -2, 1:M]) + \
                    dt * np.dot(Svec, F1[ll, :, 1:M])
            F1[0, 0:L-1] = F1[0, 1:L]
            F1[0, L-1] = func(Y2[0])
            for ll in range(iStep):  # updating stencil
                iCor = ll + 1
                F1[iCor, 0:L-1] = F1[iCor, 1:L]
                F1[iCor, L-1] = func(Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*L-2, N+L-1):
            # prediction loop
            Y2[0, :] = np.dot(A, Y2[0, :])
            # correction loops up to the second last one
            for ll in range(L-2):
                iCor = ll + 1
                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*(F1[iCor, -1, 1:M]-F1[ll, -2, 1:M]) + \
                    dt * np.dot(Svec, F1[ll, :, 1:M])
            # last correction loop
            Y2[L-1, 1:M] = Y2[L-1, 1:M] + dt * (F2[1:M]-F1[L-2, -2, 1:M]) + \
                dt * np.dot(Svec, F1[L-2, :, 1:M])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, L-1):
                F1[ll, 0:L-1] = F1[ll, 1:L]
                F1[ll, L-1] = func(Y2[ll])
            # storing the final answer
            yy[iTime+1-(L-1)] = Y2[L-1]
            F2 = func(Y2[L-1])

            F1[0, L-1] = func(Y2[0])

        return t, yy

    def heat2d_ridc(self, kappa, ics, T, N, M, L, bcs, X=(0., 1.)):
        """
        kappa: coefficient for heat equation
        ics: ics, y0
        X: spatial bounds (a=0,b=1)
        T: time bounds (0,T)
        N: no. of time intervals
        bcs: (0,0)
        M: no. of spatial nodes
        L: no. of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        """

        a, b = X
        dt = float(T)/N
        dx = (b-a)/M
        #mu = dt*kappa/dx**2
        xs = np.arange(a, b + dx, dx)
        t = np.arange(0, T+dt, dt)

        def beta(L_):
            '''
            Generates beta coefficients for Adam-Bashforth integrating scheme
            These coefficients are stored in reversed compared to conventional
            Adam-Bashforth implementations (the first element of beta corresponds to
            earlier point in time).
            input:
            M: the order of Adam-Bashforth scheme
            '''

            if L_ == 2:
                return np.array([-1./2, 3./2])
            elif L_ == 3:
                return np.array([5./12, -16./12, 23./12])
            elif L_ == 4:
                return np.array([-9./24, 37./24, -59./24, 55./24])
            elif L_ == 5:
                return np.array([251./720, -1274./720, 2616./720, -2774./720, 1901./720])
            elif L_ == 6:
                return np.array([-475./720, 2877./720, -7298./720, 9982./720, -7923./720, 4277./720])

        beta_vec = beta(4)
        beta_vec2 = beta(3)

    
        # forward Euler discretisation
        def func(u_):
            v = u_.copy()
            for i in range(1,M):
                for j in range(1, M):
                    v[i,j] = (u_[i-1,j] + u_[i+1,j]- 4*u_[i,j] +u_[i,j+1] +u_[i,j-1])/dx**2
            return v

        # Note Lm is the number of correctors
        Lm = L - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Lm, Lm+1])
        for m in range(Lm):  # Calculate qudrature weights
            for i in range(Lm+1):
                x = np.arange(Lm+1)  # Construct a polynomial
                # which equals to 1 at i, 0 at other points
                y = np.zeros(Lm+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Lm+2)
                for k in range(Lm+1):
                    P[k] = para[k]/(Lm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Lm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, M+1, M+1])
        # ics and bcs
        for i in range(1,M):
            for j in range(1,M):
                yy[0, i,j] = ics(xs[i], xs[j])
        yy[0, 0], yy[0, M], yy[0, :, 0], yy[0, :, M]  = bcs(0), bcs(0), bcs(0), bcs(0)

        # Value of RHS at initial time
        F0 = func(yy[0])
        F1 = np.zeros([Lm, L, M+1, M+1])
        F1[:, 0] = np.tile(F0,(Lm,1,1))
        F2 = F0
        # Y2 [L] new point derived in each level (prediction and corrections)
        Y2 = np.tile(yy[0],(L,1,1))

        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to L points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, L-1):
            k_1 = F1[0, iTime,:,:]
            k_2 = func(Y2[0]+(dt*4/27)*k_1 )
            k_3 = func(Y2[0]+  (dt/18)*(k_1+3*k_2))
            k_4 = func(Y2[0]+  (dt/12)*(k_1+3*k_3))
            k_5 = func(Y2[0]+   (dt/8)*(k_1+3*k_4))
            k_6 = func(Y2[0]+  (dt/54)*(13*k_1-27*k_3+42*k_4+8*k_5))
            k_7 = func(Y2[0]+(dt/4320)*(389*k_1-54*k_3+966*k_4-824*k_5+243*k_6))
            k_8 = func(Y2[0]+  (dt/20)*(-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7) )
            k_9 = func(Y2[0]+ (dt/288)*(-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8)  )
            k_10= func(Y2[0]+(dt/820)*(1481*k_1-81*k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9))
            Y2[0,1:M,1:M] = Y2[0,1:M,1:M] + dt/840*(41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10)[1:M,1:M] 
            F1[0, iTime+1] = func(Y2[0])

        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, L-1):
            ll = iCor - 1
            for iTime in range(0, L-1):
                Y2[iCor,1:M,1:M] = Y2[iCor,1:M,1:M] + dt*(F1[iCor,iTime,1:M,1:M]-F1[ll,iTime,1:M,1:M]) + \
                dt * np.tensordot(S[iTime, :],F1[ll,:,1:M,1:M],axes =1)
                F1[iCor, iTime+1] = func(Y2[iCor])  
        # treat the last correction loop a little different
        for iTime in range(0, L-1):
            Y2[L-1,1:M,1:M] = Y2[L-1,1:M,1:M] + dt*(F2[1:M,1:M]-F1[L-2, iTime,1:M,1:M]) + \
                dt * np.tensordot(S[iTime],F1[ll,:,1:M,1:M],axes =1)
            F2 = func(Y2[L-1])
            yy[iTime+1] = Y2[L-1]

        # ================== INITIAL PART (2) ==================
        beta_vec = beta(4)
        beta_vec2 = beta(3)
        for iTime in range(L-1, 2*L-2):
            iStep = iTime - (L-1)
            # prediction loop
            Y2[0,1:M,1:M] = Y2[0,1:M,1:M] + dt*np.tensordot(beta_vec,F1[0, -4:,1:M,1:M],axes =1)
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1     
                Y2[iCor,1:M,1:M] = Y2[iCor,1:M,1:M] + dt*(F1[iCor, -1,1:M,1:M]-F1[ll, -2,1:M,1:M]) + \
                    dt * np.tensordot(Svec,F1[ll,:,1:M,1:M],axes =1)
            F1[0, :L-1] = F1[0, 1:L]
            F1[0, L-1] = func(Y2[0])
            for ll in range(iStep):   #updating stencil
                iCor = ll + 1
                F1[iCor, :L-1] = F1[iCor, 1:L]
                F1[iCor, L-1] = func(Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================   
        for iTime in range(2*L-2, N+L-1):
            # prediction loop
            Y2[0,1:M,1:M] = Y2[0,1:M,1:M] + dt*np.tensordot(beta_vec,F1[0, -4:,1:M,1:M],axes =1)
            
            # correction loops up to the second last one
            for ll in range(L-2):
                iCor = ll + 1
                Fvec = np.array([F1[iCor, -3,1:M,1:M]-F1[ll, -4,1:M,1:M], F1[iCor, -2,1:M,1:M] -
                                F1[ll, -3,1:M,1:M], F1[iCor, -1,1:M,1:M]-F1[ll, -2,1:M,1:M]])            
                #print(Fvec)
                        
                Y2[iCor,1:M,1:M] = Y2[iCor,1:M,1:M] + dt*np.tensordot(beta_vec2,Fvec,axes =1) + \
                    dt * np.tensordot(Svec,F1[ll,:,1:M,1:M],axes =1)        
            # last correction loop
            F2m = func(yy[iTime+1-(L-1)-2])
            F2mm = func(yy[iTime+1-(L-1)-3])
            Fvec = np.array([F2mm[1:M,1:M]-F1[L-2, -4,1:M,1:M], F2m[1:M,1:M]-F1[L-2, -3,1:M,1:M],
                            F2[1:M,1:M]-F1[L-2, -2,1:M,1:M]]) 
            
            Y2[L-1,1:M,1:M] = Y2[L-1,1:M,1:M] + dt * np.tensordot(beta_vec2,Fvec,axes =1) + \
                dt * np.tensordot(Svec,F1[ll,:,1:M,1:M],axes =1)
            
            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, L-1):
                F1[ll, 0:L-1] = F1[ll, 1:L]
                F1[ll, L-1] = func(Y2[ll])
            # storing the final answer
            yy[iTime+1-(L-1)] = Y2[L-1]
            F2 = func(Y2[L-1])
            # ---> updating predictor stencil
            # ** approach #0:
            F1[0, :L-1] = F1[0, 1:L]
            F1[0, L-1] = func(Y2[0])

        return yy

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
    def ics(x, y): return 2*(x+y)
    #bcs = lambda x : np.sin(x)
    bcs = (0, 0), (0, 0)
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


def test3():
    def ics(_): return 10
    # 4 corrections, kappa =1, 0 at x=0 and x=1, 100 time nodes and 20 spacial ones
    t, u = Heat_fd().heat1d_ridc(
        kappa=1,
        ics=ics,
        bcs=(0, 0),
        X=(0,1),
        T=0.3,
        N=100,
        M=20,
        L=4)

    x = np.arange(0,1.05,0.05)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.axis([0,1,-10,10])
    l, = ax.plot([],[])
    def animate(i): l.set_data(x, u[i])
    ani = FuncAnimation(fig, animate, frames=100)
    HTML(ani.to_jshtml())

def test4():
    # 4 corrections, kappa =1, 0 at x=0 and x=1, 100 time nodes and 20 spacial ones
    t, u1 = Heat_fd().heat2d_ridc(
        kappa=1,
        ics=ics,
        bcs=(0, 0),
        X=(0,1),
        T=0.3,
        N=100,
        M=20,
        L=4)

    print(u1.shape)
    x = np.arange(0,1.05, 0.05)
    x2= np.linspace(0,1, 100+1)
    print(x.shape,x2.shape)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.axis([0,1,-10,10])
    l, = ax.plot([],[])
    def animate(i): l.set_data(x, u[i])
    ani = FuncAnimation(fig, animate, frames=100)
    HTML(ani.to_jshtml())


if __name__ == '__main__':
    test4()
