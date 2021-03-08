import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings

class fisher_kpp():
    def f_kpp_fe(self, alpha, beta, ics, bcs, T, N, M, X=(0., 1.)):
        """
        alpha, beta:  coefficient for u(1-u) and u_xx respectively
        ics, bcs:  u_x0, u_0t
        X: spatial bounds
        T: time bounds
        N: no. of time nodes
        M: no. of spatial nodes 
        """
        a,b = X
        dt = float(T)/N
        dx = (b-a)/M
        mu = dt/dx**2
        xs = np.arange(a, b + dx, dx)
        ts = np.arange(0, T + dt, dt)

        # u(t,x)
        u = np.zeros([N+1, M+1])
        # set bcs
        u[:, 0] = [bcs[0](t) for t in ts]
        u[:, M] = [bcs[1](t) for t in ts]
        # set ics
        u[0] = list(map(ics, xs))

        warnings.filterwarnings("ignore")
        for i_t in range(1, N):
            # Update all inner mesh points at time t[n+1]
            for i_x in range(1, M):
                u[i_t, i_x] = u[i_t-1, i_x]*(1-2*beta*mu + alpha*dt-alpha*dt*u[i_t-1, i_x]) \
                                + mu*beta*(u[i_t-1, i_x+1] + u[i_t-1, i_x-1])
        
        #! u[:,2], u[:,3], u[:,M-2], u[:,M-3] are inaccurate
        return u


    # TODO: Change to fisher_kpp ridc
    def heat_fd_ridc(self, kappa, ics, T, N, M, L, bcs=(0, 0), X=(0., 1.)):
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


def test():
    alpha, beta = 1, 1
    exact = lambda t,x : (1+np.exp(np.sqrt(alpha/6)*x -(5/6)*alpha*t))**(-2)
    #ics
    u_0x = lambda x: (1+np.exp(np.sqrt(alpha/6)*x))**(-2)
    #bcs
    u_t0 = lambda t: (1+np.exp(-(5/6)*alpha*t))**(-2)
    u_tX = lambda t: (1+np.exp(np.sqrt(alpha/6)-(5/6)*alpha*t))**(-2)

    T, N, M = 5,40,40

    u = fisher_kpp().f_kpp_fe(alpha=alpha, beta=beta, ics= u_0x, bcs=(u_t0, u_tX), T=T, N=N, M= M)

    print('numerical', u[3])

    a, b = 0, 1
    dt = float(T)/N
    dx = (b-a)/M
    xs = np.arange(a, b + dx, dx)
    ts = np.arange(0, T + dt, dt)

    U = np.zeros([N+1, M+1])

    for i in range(N+1):
        for j in range(M+1):
            U[i, j] = exact(ts[i], xs[j])

    print('analytical', U[3])


# TODO: Change to test fisher_kpp instead
def test1():
    def ics(_): return 10
    # 4 corrections, kappa =1, 0 at x=0 and x=1, 100 time nodes and 20 spacial ones
    t, u = fisher_kpp().f_kpp_ridc(
        kappa=1,
        ics=ics,
        bcs=(0, 0),
        X=(0,1),
        T=0.3,
        N=100,
        M=20,
        L=4)

    # For .py, only first 10 temporal nodes
    for i, u_i in enumerate(u):
        x = np.arange(0, 1.05, 0.05)
        fig, ax = plt.subplots()
        ax.set_title('t=' + str(t[i]))
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        ax.axis([0, 1, -10, 10])
        ax.plot(x, u_i)
        if (i == 10):
            break

    # Copy script for running on interactive python
    # x = np.arange(0,1.05,0.05)
    # fig, ax = plt.subplots()
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$u$')
    # ax.axis([0,1,-10,10])
    # ax.plot(x, u_i)
    # l, = ax.plot([],[])
    # def animate(i): l.set_data(x, u[i])
    # ani = FuncAnimation(fig, animate, frames=100)
    # HTML(ani.to_jshtml())

if __name__ == '__main__':
    test()