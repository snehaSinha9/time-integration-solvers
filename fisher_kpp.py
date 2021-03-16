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
        https://www.scirp.org/pdf/AJCM_2017040511510090.pdf
        """
        a, b = X
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

        k = alpha*dt
        R = beta*mu

        warnings.filterwarnings("ignore")
        for i_t in range(1, N):
            # Update all inner mesh points at time t[n+1]
            for i_x in range(1, M):
                u[i_t, i_x] = u[i_t-1, i_x] * \
                    (1-2*R + k - k*u[i_t-1, i_x]) + R * \
                    (u[i_t-1, i_x+1] + u[i_t-1, i_x-1])

        return u

    def f_kpp_fe2(self, a_, ics, bcs, T, N, M, X=(0., 1.)):
        """
        alpha, beta:  coefficient for u(1-u) and u_xx respectively
        ics, bcs:  u_x0, u_0t
        X: spatial bounds
        T: time bounds
        N: no. of time nodes
        M: no. of spatial nodes 
        https://www.scirp.org/pdf/AJCM_2017040511510090.pdf
        """
        a, b = X
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
                u[i_t, i_x] = u[i_t-1, i_x] + mu*(u[i_t-1, i_x-1] - 2*u[i_t-1, i_x] +
                                                  u[i_t-1, i_x+1]) + dt*u[i_t-1, i_x]*(1 - u[i_t-1, i_x])*(u[i_t-1, i_x]-a_)

        return u

    def f_kpp_ridc(self, alpha, beta, ics, T, N, M, bcs, L=4, X=(0., 1.)):
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
        xs = np.arange(a, b + dx, dx)
        ts = np.arange(0, T+dt, dt)
        k = alpha*dt

        # forward Euler discretisation
        def func(u_): 
            v = u_.copy()
            for i in range(1,M):
                v[i] = beta*(u_[i+1] - 2*u_[i] + u_[i-1])/ (dx**2) + alpha*u_[i]*(1-u_[i]) 
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
        # the final answer will be stored in yy(time, space)
        yy = np.zeros([N+1, M+1])
        # ics and bcs
        yy[0, 1:M] = np.array(list(map(ics, xs[1:M].copy())))
        yy[:, 0], yy[:, M] = [bcs[0](t) for t in ts], [bcs[1](t) for t in ts]

        # Value of RHS at initial time
        F0 = func(yy[0])
        F1 = np.zeros([Lm, L, M+1])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [L] new point derived in each level (prediction and corrections)
        Y2 = np.tile(yy[0], (L, 1))

        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to L points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, L-1):
            k_1 = F1[0, iTime]
            k_2 = func(Y2[0]+(dt*4/27)*k_1)
            k_3 = func(Y2[0] + (dt/18)*(k_1+3*k_2))
            k_4 = func(Y2[0] + (dt/12)*(k_1+3*k_3))
            k_5 = func(Y2[0] + (dt/8)*(k_1+3*k_4))
            k_6 = func(Y2[0] + (dt/54)
                       * (13*k_1-27*k_3+42*k_4+8*k_5))
            k_7 = func(Y2[0]+(dt/4320) *
                       (389*k_1-54*k_3+966*k_4-824*k_5+243*k_6))
            k_8 = func(Y2[0] + (dt/20) *
                       (-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7))
            k_9 = func(Y2[0] + (dt/288) *
                       (-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8))
            k_10 = func(Y2[0] + (dt/820)*(1481*k_1-81 *
                                          k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9))
            
            Y2[0, 1:M] = Y2[0, 1:M] + dt/840 * \
                (41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10)[1:M]
            F1[0, iTime+1] = func(Y2[0])

        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, L-1):
            ll = iCor - 1
            for iTime in range(0, L-1):
                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*(F1[iCor, iTime, 1:M]-F1[ll, iTime, 1:M]) + \
                    dt * np.tensordot(S[iTime], F1[ll, :, 1:M], axes=1)
                F1[iCor, iTime+1] = func(Y2[iCor])

        # * treat the last correction loop a little different
        for iTime in range(0, L-1):
            Y2[L-1, 1:M] = Y2[L-1, 1:M] + dt*(F2[1:M]-F1[L-2, iTime, 1:M]) + \
                dt * np.tensordot(S[iTime], F1[L-2, :, 1:M], axes=1)
            F2 = func(Y2[L-1])
            yy[iTime+1] = Y2[L-1]

        # ================== INITIAL PART (2) ==================
        def beta_(M):
            '''
            Generates beta coefficients for Adam-Bashforth integrating scheme
            These coefficients are stored in reversed compared to conventional
            Adam-Bashforth implementations (the first element of beta corresponds to
            earlier point in time).
            input:
            M: the order of Adam-Bashforth scheme
            '''

            if M == 2:
                return np.array([-1./2, 3./2])
            elif M == 3:
                return np.array([5./12, -16./12, 23./12])
            elif M == 4:
                return np.array([-9./24, 37./24, -59./24, 55./24])
            elif M == 5:
                return np.array([251./720, -1274./720, 2616./720, -2774./720, 1901./720])
            elif M == 6:
                return np.array([-475./720, 2877./720, -7298./720, 9982./720, -7923./720, 4277./720])

        beta_vec = beta_(L)
        beta_vec2 = beta_(L-1)

        for iTime in range(L-1, 2*L-2):
            iStep = iTime - (L-1)
            # prediction loop
            Y2[0, 1:M] = Y2[0, 1:M] + dt * \
                np.tensordot(beta_vec, F1[0, -4:, 1:M], axes=1)
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*(F1[iCor, -1, 1:M]-F1[ll, -2, 1:M]) + \
                    dt * np.tensordot(Svec, F1[ll, :, 1:M], axes=1)
            F1[0, 0:L-1] = F1[0, 1:L]
            
            F1[0, L-1] = func(Y2[0])
            for ll in range(iStep):  # updating stencil
                iCor = ll + 1
                F1[iCor, :L-1] = F1[iCor, 1:L]
                F1[iCor, L-1] = func(Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*L-2, N+L-1):
            # prediction loop
            Y2[0, 1:M] = Y2[0, 1:M] + dt * \
                np.tensordot(beta_vec, F1[0, -4:, 1:M], axes=1)
            # correction loops up to the second last one
            for ll in range(L-2):
                iCor = ll + 1
                Fvec = np.array([F1[iCor, -3, 1:M]-F1[ll, -4, 1:M],
                                 F1[iCor, -2, 1:M]-F1[ll, -3, 1:M],
                                 F1[iCor, -1, 1:M]-F1[ll, -2, 1:M]])

                Y2[iCor, 1:M] = Y2[iCor, 1:M] + dt*np.tensordot(beta_vec2, Fvec, axes=1) + \
                    dt * np.tensordot(Svec, F1[ll, :, 1:M], axes=1)
            # last correction loop
            Y2[L-1, 1:M] = Y2[L-1, 1:M] + dt * np.tensordot(beta_vec2, Fvec, axes=1) + \
                dt * np.tensordot(Svec, F1[L-2, :, 1:M], axes=1)

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, L-1):
                F1[ll, 0:L-1] = F1[ll, 1:L]
                F1[ll, L-1] = func(Y2[ll])
            # storing the final answer
            yy[iTime+1-(L-1)] = Y2[L-1]
            F2 = func(Y2[L-1])

            # ** approach 0
            F1[0, :L-1] = F1[0, 1:L]
            F1[0, L-1] = func(Y2[0])

        return yy



def test(T, N, M, i,j):
    # X=(a=0,b=1)
    alpha, beta = 6, 1
    def exact(t, x): return 1/np.square(1 +
                                        np.exp(np.sqrt(alpha/6)*x - (5/6)*alpha*t))
    # ics
    def u_0x(x): return 1/np.square(1+np.exp(np.sqrt(alpha/6)*x))
    # bcs
    def u_t0(t): return (1+np.exp(-(5/6)*alpha*t))**(-2)
    def u_tX(t): return (1+np.exp(np.sqrt(alpha/6)-(5/6)*alpha*t))**(-2)

    u = fisher_kpp().f_kpp_fe(alpha=alpha, beta=beta,
                              ics=u_0x, bcs=(u_t0, u_tX), T=T, N=N, M=M)

    print('numerical', u[i:j])

    a, b = 0, 1
    dt = float(T)/N
    dx = (b-a)/M
    xs = np.arange(a, b + dx, dx)
    ts = np.arange(0, T + dt, dt)

    U = np.zeros([N+1, M+1])

    for i_ in range(N+1):
        for j_ in range(M+1):
            U[i_, j_] = exact(ts[i_], xs[j_])

    print('analytical', U[i:j])


def test1(T, N, M):
    # X=(a_=0,b=1)
    a_ = 0.2

    def exact(t, x): return 0.5*(1+a_) + 0.5*(1-a_) * \
        (np.tanh(np.sqrt(2)*(1-a_)*x/4 + (1-a_**2)*t/4))
    # ics
    def u_0x(x): return 0.5*(1+a_) + 0.5*(1-a_) * \
        (np.tanh(np.sqrt(2)*(1-a_)*x/4))
    # bcs
    def u_t0(t): return 0.5*(1+a_) + 0.5*(1-a_)*(np.tanh((1-a_**2)*t/4))
    def u_tX(t): return 0.5*(1+a_) + 0.5*(1-a_) * \
        (np.tanh(np.sqrt(2)*(1-a_)/4 + (1-a_**2)*t/4))

    T, N, M = 0.01, 2000, 20

    u = fisher_kpp().f_kpp_fe(a_=a_, ics=u_0x, bcs=(u_t0, u_tX), T=T, N=N, M=M)

    print('numerical', u[:2])

    a, b = 0, 1
    dt = float(T)/N
    dx = (b-a)/M
    xs = np.arange(a, b + dx, dx)
    ts = np.arange(0, T + dt, dt)

    U = np.zeros([N+1, M+1])

    for i in range(N+1):
        for j in range(M+1):
            U[i, j] = exact(ts[i], xs[j])

    print('analytical', U[:2])

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

# TODO: Change to test fisher_kpp instead
def test_0(T, N, M, i,j):
    alpha, beta = 6, 1
    def exact(t, x): return 1/np.square(1 +
                                        np.exp(np.sqrt(alpha/6)*x - (5/6)*alpha*t))
    # ics
    def u_0x(x): return 1/np.square(1+np.exp(np.sqrt(alpha/6)*x))
    # bcs
    def u_t0(t): return (1+np.exp(-(5/6)*alpha*t))**(-2)
    def u_tX(t): return (1+np.exp(np.sqrt(alpha/6)-(5/6)*alpha*t))**(-2)

    

    # 4 corrections, kappa =1, 0 at x=0 and x=1, 100 time nodes and 20 spacial ones
    u = fisher_kpp().f_kpp_ridc(
        alpha=alpha,
        beta=beta,
        ics=u_0x,
        bcs=(u_t0, u_tX),
        X=(0, 1),
        T=T,
        N=N,
        M=M,
        L=4)
    print(u[i:j])

    # For .py, only first 10 temporal nodes
    # for i, u_i in enumerate(u):
    #     x = np.arange(0, 1.05, 0.05)
    #     fig, ax = plt.subplots()
    #     ax.set_title('t=' + str(t[i]))
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$u$')
    #     ax.axis([0, 1, -10, 10])
    #     ax.plot(x, u_i)
    #     if (i == 10):
    #         break
 
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
    T, N, M = 0.01, 100, 10
    i, j = 4, 5
    print('test pr1: T, N, M =', T, N, M)
    print('test, regular fe')
    test(T, N, M, i , j)
    #print('test 1')
    # test1()
    print('test, ridc')
    test_0(T, N, M, i, j)

