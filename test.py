from scipy.interpolate import lagrange
import numpy as np

def HOSSEINsolverFD(T, u,BC, N,L, M):
    '''
    RIDC solver for 1D heat equation
    
    Inputs:
    T:  integration interval[0,T]
    y0: initial condition
    BC: boundary conditions
    N:  number of nodes
    L: number of spacial nodes
    M: the number of points in calculating quadraure integral
    (and also the number of steps used in Adam-Bashforth predictor)
    or number of correction loops PLUS the prection loop

    Output:
    t: time vector
    yy: solution as a function of time
    '''

    # time step and spacial step
    h = float(T)/N
    dx = 1./L
    mu = h/dx**2
    A = np.diagflat((1-2*mu)*np.ones(L+1))+np.diagflat((mu)*np.ones(L),1)+np.diagflat((mu)*np.ones(L),-1)
    A[0,0]=1
    A[-1,-1]=1
    A[0,1]=0
    A[-1,-2]=0
    
    ##### define the semi-discrete system RHS, we will hardcode it for heat eq first (maybe?)
    
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
    
    # the time vector
    t = np.arange(0, T+h, h)
    # extended time vector (temporary: cuz I didn't write code for end part)
    t_ext = np.arange(0, T+h+M*h, h)
    # the final answer will be stored in yy

    yy = u
    # Value of RHS at initial time
    F0 = func(yy[0])

    # F vector and matrice:
    # the RHS of ODE is evaluated and stored in this vector and matrix:
    # F1 [M x M]: first index is the order (0=prection, 1=first correction)
    # second index is the time (iTime)
    # Note F1 could have been [M-1 x M] as the first two rows are equal to each
    # other BUT we designed it as a place holder for future parallelisation
    F1 = np.zeros([Mm, M,L+1])
    F1[:, 0] = np.array([F0]*Mm).T
    F2 = F0
    Y2 = np.array([yy[0]]*M).T
    
    
    # ================== INITIAL PART (1) ==================
    # for this part the predictor and correctors step up to M points in time
    # ** predictor ** uses Runge-Kutta 4
    for iTime in range(0, M-1):

        Y2[0,:] = np.dot(A,Y2[0,:])
        F1[0, iTime+1,:] = func(Y2[0,:])
    
    # ** correctors ** use Integral Deffered Correction
    for iCor in range(1, M-1):
        ll = iCor - 1
        for iTime in range(0, M-1):
            Y2[iCor,1:L] = Y2[iCor,1:L] + h*(F1[iCor,iTime,1:L]-F1[ll,iTime,1:L]) + \
            h * np.dot(S[iTime, :], F1[ll, :,1:L])
            F1[iCor, iTime+1,:] = func(Y2[iCor,:])
        
    # treat the last correction loop a little different
    for iTime in range(0, M-1):
        Y2[M-1,1:L] = Y2[M-1,1:L] + h*(F2[1:L]-F1[M-2, iTime,1:L]) + \
            h * np.dot(S[iTime, :], F1[M-2, :,1:L])
        F2 = func(Y2[M-1,:])
        yy[iTime+1,:] = Y2[M-1,:]

    # ================== INITIAL PART (2) ==================
    
    for iTime in range(M-1, 2*M-2):
        iStep = iTime - (M-1)
        # prediction loop
        Y2[0,:] = np.dot(A,Y2[0,:])
        # correction loops
        for ll in range(iStep):
            iCor = ll + 1
            Y2[iCor,1:L] = Y2[iCor,1:L] + h*(F1[iCor, -1,1:L]-F1[ll, -2,1:L]) + \
                h * np.dot(Svec, F1[ll, :,1:L])
        F1[0, 0:M-1,:] = F1[0, 1:M,:]
        F1[0, M-1,:] = func(Y2[0,:])
        for ll in range(iStep):   #updating stencil
            iCor = ll + 1
            F1[iCor, 0:M-1,:] = F1[iCor, 1:M,:]
            F1[iCor, M-1,:] = func(Y2[iCor,:])

    # ================== MAIN LOOP FOR TIME ==================
    for iTime in range(2*M-2, N+M-1):
        # prediction loop
        Y2[0,:] = np.dot(A,Y2[0,:])
        # correction loops up to the second last one
        for ll in range(M-2):
            iCor = ll + 1
            Y2[iCor,1:L] = Y2[iCor,1:L] + h*(F1[iCor, -1,1:L]-F1[ll, -2,1:L]) + \
                h * np.dot(Svec, F1[ll, :,1:L])
        # last correction loop
        Y2[M-1,1:L] = Y2[M-1,1:L] + h * (F2[1:L]-F1[M-2, -2,1:L]) + \
            h * np.dot(Svec, F1[M-2, :,1:L])
        
        # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
        # ---> updating correctors stencil
        for ll in range(1, M-1):
            F1[ll, 0:M-1,:] = F1[ll, 1:M,:]
            F1[ll, M-1,:] = func(Y2[ll,:])
        # storing the final answer
        yy[iTime+1-(M-1),:] = Y2[M-1,:]
        F2 = func(Y2[M-1,:])


        F1[0, M-1,:] = func(Y2[0,:])

    return t, yy