'''
Analysing RIDC parameters for predictor and correctors
'''

from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from functools import reduce
from scipy.integrate import quadrature
import numpy as np
from functools import wraps
from time import time
import test_examples as ex


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        return result, end-start
    return wrapper


class DCs:

    @timing
    def idc_fe(self, a, b, alpha, N, p, f):
        """Perform IDCp-FE
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; f vector field.
        Require: N divisible by M=p-1, with JM=N. M is #corrections.
        Return: eta_sol
        """

        # Initialise, J intervals of size M
        if not type(N) is int:
            raise TypeError('N must be integer')
        M = p-1
        if N % M != 0:
            raise Exception('p-1 does not divide N')
        dt = (b-a)/N
        J = int(N/M)
        S = np.zeros([M, M+1])

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros(N+1)
        eta_sol[0] = alpha
        eta = np.zeros([M+1, J, M+1])
        t = np.zeros([J, M+1])
        eta_overlaps = np.zeros([J])

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]

        for j in range(J):
            # Prediction Loop
            eta[0, j, 0] = alpha if j == 0 else eta_overlaps[j]
            for m in range(M):
                t[j, m] = (j*M+m)*dt
                eta[0, j, m+1] = eta[0, j, m] + dt*f(t[j, m], eta[0, j, m])

            # Correction Loops
            for l in range(1, M+1):
                eta[l, j, 0] = eta[l-1, j, 0]
                for m in range(M):
                    # Error equation, Forward Euler
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt * \
                        np.sum([S[m, i] * f(t[j, i], eta[l-1, j, i])
                                for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2

            eta_sol[j*M+1:(j+1)*M + 1] = eta[M, j, 1:]
            if j != J-1:
                eta_overlaps[j+1] = eta[M, j, M]

        return eta_sol[:-1]

    def idc_stability(self, l, p, testeq):
        # Author: Sam Brossler
        # l - lambda value, p - order so M=p-1
        M = p-1
        J = 1
        sol_list = np.zeros(J*M+1, dtype=complex)
        sol_list[0] = 1.
        Y = np.zeros((J, M+1), dtype=complex)  # approx solution
        Y1 = np.zeros((J, M+1), dtype=complex)  # corrected solution
        Y[0, 0] = 1  # inital value
        S = np.zeros((M, M+1))  # integration matrix

        for m in range(M):   # calculating integration matrix
            for i in range(M+1):
                x = np.arange(M+1)  # Construct a polynomial
                y = np.zeros(M+1)   # which equals 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)  # constructs polynomial
                para = np.poly1d.integ(p)
                # finds definite integral of polynomial and adds to integral matrix
                S[m, i] = para(m+1) - para(m)

        for j in range(J):
            for m in range(M):  # prediction
                # Eulers forward method
                Y[j, m+1] = Y[j, m] + testeq(l, Y[j, m])
            for _ in range(1, M+1):  # correction
                Y1[j, 0] = Y[j, 0]

                for m in range(M):
                    g = 0
                    for k in range(M+1):
                        g = g + S[m, k]*testeq(l, Y[j, k])
                    # solve error equation with forward Euler
                    Y1[j, m+1] = Y1[j, m] + \
                        (testeq(l, Y[j, m])-testeq(l, Y1[j, m])) + g
                Y[j, :] = Y1[j, :]
            sol_list[j*M+1:j*M+M+1] = Y1[j, 1:M+1]
            if j != J-1:  # set initial value for next iteration
                Y[j+1, 0] = Y1[j, M]

        return sol_list

    @timing
    def ridc_fe(self, a, b, alpha, N, p, K, f):
        """Perform IDCp-FE
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=N. M is
        Return: eta_sol
        """

        # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])
        d = 1 if isinstance(alpha, int) else len(alpha)

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros([N+1, d], dtype=complex)
        eta_sol[0] = alpha
        eta0 = np.zeros([J, K+1, d], dtype=complex)
        eta1 = np.zeros([J, K+1, d], dtype=complex)
        t = np.zeros([J, K+1])

        times = np.linspace(a, b, N+1)
        for j in range(J):
            t[j] = times[j*K:(j+1)*K+1]

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                x = np.arange(M+1)
                y = np.zeros(M+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.poly1d.integ(p)
                S[m, i] = para(m+1) - para(m)

        for j in range(J):
            # predictor starts w last point in j-1 interval
            eta0[:, 0] = eta_sol[j*K]
            for m in range(K):  # prediction
                eta0[j, m+1] = eta0[j, m] + dt * \
                    f(t[j, m], eta0[j, m])  # Eulers forward method

            for _ in range(1, M+1):  # correction
                eta1[j, 0] = eta0[j, 0]

                for m in range(M):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt*sum([S[m, k]*f(t[j, k], eta0[j, k])
                                    for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt * \
                        sum([S[M-1, k]*f(t[j, m-M+k+1], eta0[j, m-M+k+1])
                             for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                eta0[j] = eta1[j]

            eta_sol[j*K+1:j*K+K+1] = eta1[j, 1:]
        return eta_sol

    def ridc_fe_stab(self,p,K,l, testeq):#Euler
        #(a,b)-endpoints, N-number of steps, p-order of method, K- No. intervals,  y0-I.C, F-function    
        M = p-1  
        J = int(10/K)
        sol_list = np.zeros(11,dtype = complex) #stores the solution N+1
        sol_list[0] = 1.
        Y = np.zeros((J,K+1),dtype = complex)  #approx solution
        Y1 = np.zeros((J,K+1),dtype = complex)   #corrected solution
        Y[0,0]= 1.           #inital value
        S = np.zeros((M,M+1))  #integration matrix
        
        for m in range(M):   # calculating integration matrix
            for i in range(M+1):
                x = np.arange(M+1)  # Construct a polynomial
                y = np.zeros(M+1)   # which equals 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)  # constructs polynomial
                para = np.poly1d.integ(p)  
                S[m,i] = para(m+1) - para(m)  #finds definite integral of polynomial and adds to integral matrix
                
        for j in range(J):
            Y[:, 0] = sol_list[j*K]  # predictor starts w last point in j-1 interval
            for m in range(K):   #prediction
                Y[j,m+1] = Y[j,m] + testeq(l,Y[j,m]) #Eulers forward method  
                
            for _ in range(1,M+1):   #correction
                Y1[j,0] = Y[j,0]
                
                for m in range(M):  
                    
                    g = sum([S[m,k]*testeq(l,Y[j,k]) for k in range(M+1)])

                    Y1[j,m+1] = Y1[j,m] + testeq(l,Y1[j,m])-testeq(l,Y[j,m]) + g #solve error equation with forward Euler
                
                for m in range(M,K):
                    
                    g = sum([S[M-1,k]*testeq(l,Y[j,m-M+k+1]) for k in range(M+1)])
                    
                    Y1[j,m+1] = Y1[j,m] + (testeq(l,Y1[j,m])-testeq(l,Y[j,m])) + g  #solve error equation with forward Euler
                        
                Y[j,:] = Y1[j,:]
        
            sol_list[j*K+1:j*K+K+1] = Y1[j,1:K+1]

                
        return sol_list

    @timing
    def ridc_rk4(self, a, b, alpha, N, p, K, f):
        """Perform RIDC(p,K)-RK(4)
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=M
        Return: eta_sol
        """

       # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])
        d = 1 if isinstance(alpha, int) else len(alpha)

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros([N+1, d])
        eta_sol[0] = alpha
        eta0 = np.zeros([J, K+1, d])
        eta1 = np.zeros([J, K+1, d])
        t = np.zeros([J, K+1])

        times = np.linspace(a, b, N+1)
        for j in range(J):
            t[j] = times[j*K:(j+1)*K+1]

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                x = np.arange(M+1)
                y = np.zeros(M+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.poly1d.integ(p)
                S[m, i] = para(m+1) - para(m)

        for j in range(J):
            # predictor starts w last point in j-1 interval
            eta0[:, 0] = eta_sol[j*K]
            for m in range(K):  # prediction
                k1 = f(t[j, m], eta0[j, m])
                k2 = f(t[j, m]+dt/2, eta0[j, m] + k1/2)
                k3 = f(t[j, m]+dt/2, eta0[j, m]+k2/2)
                k4 = f(t[j, m]+dt, eta0[j, m]+k3)
                eta0[j, m+1] = eta0[j, m] + dt * \
                    (k1/6 + k2/3 + k3/3 + k4/6)  # RK4

            for _ in range(1, M+1):  # correction
                eta1[j, 0] = eta0[j, 0]

                for m in range(M):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt*sum([S[m, k]*f(t[j, k], eta0[j, k])
                                    for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt * \
                        sum([S[M-1, k]*f(t[j, m-M+k+1], eta0[j, m-M+k+1])
                             for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                eta0[j] = eta1[j]

            eta_sol[j*K+1:j*K+K+1] = eta1[j, 1:]
        return eta_sol

    @timing
    def ridc_rk8(self, T, y0, N, M, f):
        """Perform RIDC(p,K)-RK(4)
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=M
        Return: eta_sol
        """

       # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        K = 1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = T/N
        J = int(N/K)
        S = np.zeros([M, M+1])
        d = 1 if isinstance(y0, int) else len(y0)

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros([N+1, d])
        eta_sol[0] = y0
        eta0 = np.zeros([J, K+1, d])
        eta1 = np.zeros([J, K+1, d])
        t = np.zeros([J, K+1])

        times = np.linspace(0, T, N+1)
        for j in range(J):
            t[j] = times[j*K:(j+1)*K+1]

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                x = np.arange(M+1)
                y = np.zeros(M+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.poly1d.integ(p)
                S[m, i] = para(m+1) - para(m)

        for j in range(J):
            # predictor starts w last point in j-1 interval
            eta0[:, 0] = eta_sol[j*K]
            for m in range(K):  # prediction
                k1 = f(t[j, m], eta0[j, m])
                k2 = f(t[j, m]+dt/7, eta0[j, m] + dt*k1/7)
                k3 = f(t[j, m]+2*dt/7, eta0[j, m]+dt/1323 *(7538*k1 -7160*k2))
                k4 = f(t[j, m]+3*dt/7, eta0[j, m]+dt/5978 *(549*k1 +4882*k2-2869*k3))
                k5 = f(t[j, m]+4*dt/7, eta0[j, m]+dt/427 *(-693*k1 +682*k2-211*k3 + 466*k4))
                k6 = f(t[j, m]+5*dt/7, eta0[j, m]+dt/378 *(-79*k1 +322*k2+244*k3 + 126*k4 -323*k5))
                k7 = f(t[j, m]+6*dt/7, eta0[j, m]+dt/3577 *(-2537*k1 +2568*k2+1021*k3 + 511*k4 + 511*k5 + 992*k6))
                k8 = f(t[j, m]+dt, eta0[j, m]+dt/1502 *(-61*k1 +102*k2+428*k3 -112*k4 + 126*k5 + 242*k6 +777*k7))
                eta0[j, m+1] = eta0[j, m] + dt/120960 * \
                    (5257*k1 +25039*k2+9261*k3+20923*k4 + 20923*k5 + 9261*k6 +25039*k7 + 5257*k8)  # RK4

            for _ in range(1, M+1):  # correction
                eta1[j, 0] = eta0[j, 0]

                for m in range(M):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt*sum([S[m, k]*f(t[j, k], eta0[j, k])
                                    for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt * \
                        sum([S[M-1, k]*f(t[j, m-M+k+1], eta0[j, m-M+k+1])
                             for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                eta0[j] = eta1[j]

            eta_sol[j*K+1:j*K+K+1] = eta1[j, 1:]
        return eta_sol

    @timing
    def ridc_ab2(self, a, b, alpha, N, p, K, f):
        """Perform RIDC(p,K)-AB2
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=N. M is
        Return: eta_sol
        """
        # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])
        d = 1 if isinstance(alpha, int) else len(alpha)

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros([N+1, d])
        eta_sol[0] = alpha
        eta0 = np.zeros([J, K+1, d])
        eta1 = np.zeros([J, K+1, d])
        t = np.zeros([J, K+1])

        times = np.linspace(a, b, N+1)
        for j in range(J):
            t[j] = times[j*K:(j+1)*K+1]

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                x = np.arange(M+1)
                y = np.zeros(M+1)
                y[i] = 1
                p = lagrange(x, y)
                para = np.poly1d.integ(p)
                S[m, i] = para(m+1) - para(m)

        for j in range(J):
            # predictor starts w last point in j-1 interval
            eta0[:, 0] = eta_sol[j*K]
            eta0[j, 1] = eta0[j, 0] + dt*f(t[j, 0], eta0[j, 0])
            for m in range(K-1):  # prediction
                eta0[j, m+2] = eta0[j, m+1] + 1.5*dt * \
                    f(t[j, m+1], eta0[j, m+1]) - 0.5*dt*f(t[0, m], eta0[j, m])

            for _ in range(1, M+1):  # correction
                eta1[j, 0] = eta0[j, 0]

                for m in range(M):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt*sum([S[m, k]*f(t[j, k], eta0[j, k])
                                    for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta1[j, m])-f(t[j, m], eta0[j, m]))
                    term2 = dt * \
                        sum([S[M-1, k]*f(t[j, m-M+k+1], eta0[j, m-M+k+1])
                             for k in range(M+1)])
                    # solve error equation with forward Euler
                    eta1[j, m+1] = eta1[j, m] + term1 + term2

                eta0[j] = eta1[j]

            eta_sol[j*K+1:j*K+K+1] = eta1[j, 1:]

        return eta_sol

    def ridc_abM(self, T, y0, N, M, approach, f):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        '''

        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        d = len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        # Precompute integration matrix
        for m in range(Mm):  # Calculate qudrature weights
            for i in range(Mm+1):
                x = np.arange(Mm+1)  # Construct a polynomial
                y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Mm+2)
                for k in range(Mm+1):
                    P[k] = para[k]/(Mm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Mm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, d])
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = f(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M, d])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones([M, d])*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            KK1 = F1[0, iTime]
            KK2 = f(t[iTime]+h/2, Y2[0]+KK1*h/2)
            KK3 = f(t[iTime]+h/2, Y2[0]+KK2*h/2)
            KK4 = f(t[iTime]+h,   Y2[0]+KK3*h)
            Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
            F1[0, iTime+1] = f(t[iTime+1], Y2[0])
        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime], F1[ll])
                F1[iCor, iTime+1] = f(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime], F1[M-2])
            F2 = f(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
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

        beta_vec = beta(M)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll])
            F1[0, 0:M-1] = F1[0, 1:M]
            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0:M-1] = F1[iCor, 1:M]
                F1[iCor, M-1] = f(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # correction loops up to the second last one
            for ll in range(M-2):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll, :])
            # last correction loop
            Y2[M-1] = Y2[M-1] + h * (F2-F1[M-2, -2]) + \
                h * np.dot(Svec, F1[M-2, :])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0:M-1] = F1[ll, 1:M]
                F1[ll, M-1] = f(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = f(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ** updating predictor stencil
            if (approach == 0):
                F1[0, 0:M-1] = F1[0, 1:M]
            # ** approach #1: pushing the most correct answer to predictor
            elif (approach == 1):
                F1[0, 0] = F2
                F1[0, 1:M-1] = F1[0, 2:M]
            # ** approach #2 : pushing the recently corrected answer of
            else:
                # each corrector to the associated node in predictor
                F1[0, 0] = F2
                for ii in range(1, M-1):
                    F1[0, ii] = F1[-ii, -1]

            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])

        return yy

    def ridc_abM2(self, T, y0, N, M, approach, f):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        '''

        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        d = len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        # Precompute integration matrix
        for m in range(Mm):  # Calculate qudrature weights
            for i in range(Mm+1):
                x = np.arange(Mm+1)  # Construct a polynomial
                y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Mm+2)
                for k in range(Mm+1):
                    P[k] = para[k]/(Mm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Mm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, d])
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = f(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M, d])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones([M, d])*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            k_1 = F1[0, iTime]
            k_2 = f(t[iTime]+h*(4/27),Y2[0]+(h*4/27)*k_1 )
            k_3 = f(t[iTime]+h*(2/9) ,Y2[0]+  (h/18)*(k_1+3*k_2))
            k_4 = f(t[iTime]+h*(1/3) ,Y2[0]+  (h/12)*(k_1+3*k_3))
            k_5 = f(t[iTime]+h*(1/2) ,Y2[0]+   (h/8)*(k_1+3*k_4))
            k_6 = f(t[iTime]+h*(2/3) ,Y2[0]+  (h/54)*(13*k_1-27*k_3+42*k_4+8*k_5))
            k_7 = f(t[iTime]+h*(1/6) ,Y2[0]+(h/4320)*(389*k_1-54*k_3+966*k_4-824*k_5+243*k_6))
            k_8 = f(t[iTime]+h       ,Y2[0]+  (h/20)*(-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7) )
            k_9 = f(t[iTime]+h*(5/6) ,Y2[0]+ (h/288)*(-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8)  )
            k_10= f(t[iTime]+h       ,Y2[0]+(h/820)*(1481*k_1-81*k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9))
            Y2[0] = Y2[0] + h/840*(41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10)
            
            
            F1[0, iTime+1] = f(t[iTime+1], Y2[0])
        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime], F1[ll])
                F1[iCor, iTime+1] = f(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime], F1[M-2])
            F2 = f(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
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

        beta_vec = beta(M)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll])
            F1[0, 0:M-1] = F1[0, 1:M]
            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0:M-1] = F1[iCor, 1:M]
                F1[iCor, M-1] = f(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # correction loops up to the second last one
            for ll in range(M-2):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll, :])
            # last correction loop
            Y2[M-1] = Y2[M-1] + h * (F2-F1[M-2, -2]) + \
                h * np.dot(Svec, F1[M-2, :])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0:M-1] = F1[ll, 1:M]
                F1[ll, M-1] = f(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = f(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ** updating predictor stencil
            if (approach == 0):
                F1[0, 0:M-1] = F1[0, 1:M]
            # ** approach #1: pushing the most correct answer to predictor
            elif (approach == 1):
                F1[0, 0] = F2
                F1[0, 1:M-1] = F1[0, 2:M]
            # ** approach #2 : pushing the recently corrected answer of
            else:
                # each corrector to the associated node in predictor
                F1[0, 0] = F2
                for ii in range(1, M-1):
                    F1[0, ii] = F1[-ii, -1]

            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])

        return yy

    #! AUTHOR: Hossein and Tianming (??) original
    def ridc_hosseinAB(self, func, T, y0, N, M):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop
        Output:
        t: time vector
        yy: solution as a function of time
        '''
        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        # d = 1  # len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        for m in range(Mm):  # Calculate qudrature weights
            for i in range(Mm+1):
                x = np.arange(Mm+1)  # Construct a polynomial
                y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Mm+2)
                for k in range(Mm+1):
                    P[k] = para[k]/(Mm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Mm-1, :]
        # the final answer will be stored in yy
        yy = np.zeros(N+1)
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = func(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones(M)*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            KK1 = F1[0, iTime]
            KK2 = func(t[iTime]+h/2, Y2[0]+KK1*h/2)
            KK3 = func(t[iTime]+h/2, Y2[0]+KK2*h/2)
            KK4 = func(t[iTime]+h,   Y2[0]+KK3*h)
            Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
            F1[0, iTime+1] = func(t[iTime+1], Y2[0])
        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime, :], F1[ll, :])
                F1[iCor, iTime+1] = func(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime, :], F1[M-2, :])
            F2 = func(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
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

        beta_vec = beta(M)
        beta_vec2 = beta(M-1)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll, :])
            F1[0, 0: M-1] = F1[0, 1: M]
            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0: M-1] = F1[iCor, 1: M]
                F1[iCor, M-1] = func(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # correction loops up to the second last one
            for ll in range(M-2):
                iCor = ll + 1
                Fvec = np.array([F1[iCor, -3]-F1[ll, -4], F1[iCor, -2] -
                                F1[ll, -3], F1[iCor, -1]-F1[ll, -2]])
                Y2[iCor] = Y2[iCor] + h*np.dot(beta_vec2, Fvec) + \
                    h * np.dot(Svec, F1[ll, :])
            # last correction loop
            F2m = func(t_ext[iTime+1-(M-1)-2], yy[iTime+1-(M-1)-2])
            F2mm = func(t_ext[iTime+1-(M-1)-3], yy[iTime+1-(M-1)-3])
            Fvec = np.array([F2mm-F1[M-2, -4], F2m-F1[M-2, -3], F2-F1[M-2, -2]])
            Y2[M-1] = Y2[M-1] + h*np.dot(beta_vec2, Fvec) + \
                h * np.dot(Svec, F1[M-2, :])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0: M-1] = F1[ll, 1: M]
                F1[ll, M-1] = func(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = func(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ---> updating predictor stencil
            # ** approach #0:
            F1[0, 0: M-1] = F1[0, 1: M]
            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

        return t, yy


    def ridc_hossein_test1(self, func, T, y0, N, M):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop
        Output:
        t: time vector
        yy: solution as a function of time
        '''
        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        # d = 1  # len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        for m in range(Mm):  # Calculate qudrature weights
            for i in range(Mm+1):
                x = np.arange(Mm+1)  # Construct a polynomial
                y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
                y[i] = 1
                p = lagrange(x, y)
                para = np.array(p)    # Compute its integral
                P = np.zeros(Mm+2)
                for k in range(Mm+1):
                    P[k] = para[k]/(Mm+1-k)
                P = np.poly1d(P)
                S[m, i] = P(m+1) - P(m)
        Svec = S[Mm-1]
        # the final answer will be stored in yy
        yy = np.zeros(N+1)
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = func(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones(M)*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            k_1 = F1[0, iTime]
            k_2 = func(t[iTime]+h*(4/27),Y2[0]+(h*4/27)*k_1 )
            k_3 = func(t[iTime]+h*(2/9) ,Y2[0]+  (h/18)*(k_1+3*k_2))
            k_4 = func(t[iTime]+h*(1/3) ,Y2[0]+  (h/12)*(k_1+3*k_3))
            k_5 = func(t[iTime]+h*(1/2) ,Y2[0]+   (h/8)*(k_1+3*k_4))
            k_6 = func(t[iTime]+h*(2/3) ,Y2[0]+  (h/54)*(13*k_1-27*k_3+42*k_4+8*k_5))
            k_7 = func(t[iTime]+h*(1/6) ,Y2[0]+(h/4320)*(389*k_1-54*k_3+966*k_4-824*k_5+243*k_6))
            k_8 = func(t[iTime]+h       ,Y2[0]+  (h/20)*(-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7) )
            k_9 = func(t[iTime]+h*(5/6) ,Y2[0]+ (h/288)*(-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8)  )
            k_10= func(t[iTime]+h       ,Y2[0]+(h/820)*(1481*k_1-81*k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9))
            Y2[0] = Y2[0] + h/840*(41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10)
            F1[0, iTime+1] = func(t[iTime+1], Y2[0])


        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime], F1[ll])
                F1[iCor, iTime+1] = func(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime], F1[M-2])
            F2 = func(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
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

        beta_vec = beta(M)
        beta_vec2 = beta(M-1)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
 
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll])
            F1[0, 0: M-1] = F1[0, 1: M]
            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0: M-1] = F1[iCor, 1: M]
                F1[iCor, M-1] = func(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # prediction loop
            # print(Y2[0])
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
            # correction loops up to the second last one
            for ll in range(M-2):
                iCor = ll + 1
                Fvec = np.array([F1[iCor, -3]-F1[ll, -4], F1[iCor, -2] -
                                F1[ll, -3], F1[iCor, -1]-F1[ll, -2]])
                Y2[iCor] = Y2[iCor] + h*np.dot(beta_vec2, Fvec) + \
                    h * np.dot(Svec, F1[ll])
            # last correction loop
            F2m = func(t_ext[iTime+1-(M-1)-2], yy[iTime+1-(M-1)-2])
            F2mm = func(t_ext[iTime+1-(M-1)-3], yy[iTime+1-(M-1)-3])
            Fvec = np.array([F2mm-F1[M-2, -4], F2m-F1[M-2, -3], F2-F1[M-2, -2]])
            Y2[M-1] = Y2[M-1] + h*np.dot(beta_vec2, Fvec) + \
                h * np.dot(Svec, F1[M-2])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0: M-1] = F1[ll, 1: M]
                F1[ll, M-1] = func(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = func(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ---> updating predictor stencil
            # ** approach #0:
            F1[0, 0: M-1] = F1[0, 1: M]
            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

        return t, yy