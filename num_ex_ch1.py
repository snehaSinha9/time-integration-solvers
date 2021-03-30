# chapter 1 graphs
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame as DF

from scipy.interpolate import lagrange
import test_examples as ex
from dc_experiments import DCs as dc
from serial_examples import Schemes as scheme


def orders():
    y0 = 1
    y = ex.y_exact0
    dy_dt = ex.func0
    def analyse_scheme(dy_dt, y, a, b, ics, step_sizes, order_idc):
        max_global_err = []
        for i in range(len(step_sizes)):
            h = step_sizes[i]
            y_pred = scheme(v_field=dy_dt,
                            start=a, stop=b, h=h, init_conditions=ics)

            times = np.arange(a, b, h)
            exact = [y(t) for t in times]
            
            df = DF({'times': times,
                        'euler': y_pred.euler()[1],
                        'rk4':  y_pred.rk4()[1],
                        'ab4': y_pred.ab4()[1],
                        'exact': exact})
            idc_err = []
            for order in order_idc:
                ts, df['idc_' + str(order)] = dc().idc_fe(a=a, b=b, alpha=y0, N=int((b-a)/h), p=order, f=dy_dt)
                df['exact_'+str(order)] = [y(t) for t in ts]
                idc_err = idc_err + [df['exact_'+str(order)].sub(df['idc_' + str(order)]).abs().max()]
                
            df['error_euler'] = df['exact'].sub(df['euler']).abs()
            df['error_rk4'] = df['exact'].sub(df['rk4']).abs()
            df['error_ab4'] = df['exact'].sub(df['ab4']).abs()
            max_global_err.append(idc_err+[df['error_euler'].max(),
                                        df['error_rk4'].max(),
                                        df['error_ab4'].max()])

        return max_global_err

    h_sizes = [10**i for i in [-1, -2, -3, -4]]

    start, stop = 0, 1

    N = [1/h for h in h_sizes]

    max_global_err = analyse_scheme(dy_dt, y, start, stop, y0, h_sizes, [3,6,11])

    plt.plot(N, max_global_err)

    plt.plot(N[:-1],max_global_err[:-1])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('log Maximum Global Error')
    plt.xlabel('log Number of time-steps')
    plt.legend(['IDC(3)-FE', 'IDC(6)-FE', 'IDC(11)-FE','Forward Euler (order 1)', 'Runge-Kutta (order 4)','Adam Bashforth (order 4)', ], loc='centre right', bbox_to_anchor=(1, 0.7))
    plt.show()

# 1 step
def stability_1step(col, R, axisbox=[-2, 2, -2, 2], method='.'):
    nptsx = 501
    nptsy = 501
    xa, xb, ya, yb = axisbox
    x = np.linspace(xa, xb, nptsx)
    y = np.linspace(ya, yb, nptsy)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    Rabs = abs(R(Z))
    print(Rabs.shape)
    print(X.shape, Y.shape)
    levels = [-1e9, 1, 1e9]
    Sregion_color = [0.8, 0.8, 1.]   # RGB
    # plt.contourf(X, Y, Rabs, levels, colors=[Sregion_color, 'w'])
    l=plt.contour(X, Y, Rabs, [1], colors=col)  # boundary
    #plt.clabel(l, inline=1, fontsize=10)
    print(l.labels)
    # plot axes
    plt.plot([xa, xb], [0, 0], 'k')
    plt.plot([0, 0], [ya, yb], 'k')
    plt.axis(axisbox)
    plt.axis('scaled')  # so circles are circular
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')
    return l

# multi-step
def stability_mstep(rho, sigma, axisbox=[-2, 2, -2, 2], method='.'):
    theta = np.linspace(0, 2*np.pi, 1000)
    eitheta = np.exp(1j * theta)
    z = rho(eitheta) / sigma(eitheta)
    l = plt.plot(z.real, z.imag, 'r', linewidth=2)
    # plot axes
    xa, xb, ya, yb = axisbox
    plt.plot([xa, xb], [0, 0], 'k')
    plt.plot([0, 0], [ya, yb], 'k')
    plt.axis(axisbox)
    plt.axis('scaled')  # so circles are circular
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')

    return l


def tests(i):
    if (i == 1):
        # AB4
        def rho(z): return (z-1.) * z**3
        def sigma(z): return (((55*z - 59)*z + 37.)*z - 9) / 24.
        return stability_mstep(rho, sigma, method='AB4')
    elif (i == 2):
        # BE
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1./(1.-z)
        return stability_1step('c', R, [-2.5, 2.5, -2, 2], 'BE')
    elif (i == 3):
        # FE
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1./(1.+z)
        return stability_1step('b', R, [-2.5, 2.5, -2, 2], 'FE')
    elif (i == 4):
        # CN
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return (1. + z/2.)/(1.-z/2.)
        return stability_1step('m',R, [-2.5, 2.5, -2, 2], 'CN')
    elif (i == 5):
        # RK4
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
        return stability_1step('g', R, [-5, 5, -5, 5], 'RK4')
    elif (i == 6):
        # Implicit trapozodial
        def rho(z): return z-1.
        def sigma(z): return z/2. + 0.5
        return stability_mstep(rho, sigma, [-2, 2, -2, 2], 'Implicit CN')
    elif (i == 7):
        # Implicit trapozodial
        def rho(z): return z-1.
        def sigma(z): return 1
        return stability_mstep(rho, sigma, [-2, 2, -2, 2])

# !stability_dc, does not work

def stability_dc(axisbox=[-2, 2, -2, 2]):
    # idc-stability
    # ! Does not work. do not attempt to use
    nptsx = 10
    nptsy = 10
    axisbox = [-5, 0, -5, 5]
    xa, xb, ya, yb = axisbox
    x = np.linspace(xa, xb, nptsx)
    y = np.linspace(ya, yb, nptsy)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    stab_func = ex.stab_eq
    K = [5]
    for k in K:
        def r(z):
            def func_lambda(t, y): return stab_func(z, y)
            return abs(max(dc().ridc_fe(0, 1, 1, 20, k+1, k, func_lambda)[0]))

        Rabs = np.array([[r(zx) for zx in zy] for zy in Z])[:, :, 0]
        print(Y.shape, Y.shape, Rabs.shape)
        levels = [-1e9, 1, 1e9]
        Sregion_color = [0.8, 0.8, 1.]   # RGB
        plt.contourf(X, Y, Rabs, levels, colors=[Sregion_color, 'w'])

    # plot axes
    plt.plot([xa, xb], [0, 0], 'k')
    plt.plot([0, 0], [ya, yb], 'k')
    plt.axis(axisbox)
    plt.axis('scaled')  # so circles are circular
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')
    plt.show()

    # testing for IDC


def idc_stab(M):
    # *Author: Sam Brosler
    f = ex.stab_eq
    col = ['y','g','c','m']
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for I, m in enumerate(M):
        reals_in_range = []
        imgs_in_range = []
        # Finding the real values that for the stability interval
        for i in np.arange(0, 3.01, 0.01):
            reals_p = dc().idc_stability(-i, m, f)
            if abs(reals_p[-1]) <= 1.:
                reals_in_range.append(round(-i, 6))
                print(i)
            else:
                break

        # finding the maximum imaginary values correcsponding to 
        # the real ones that satisfy the amplification factor
        for r in reals_in_range:
            goodlistIm1 = []
            for i in np.arange(0, 3.01, 0.01):
                ims_p = dc().idc_stability(complex(r, i), m, f)
                if abs(ims_p[-1]) <= 1.:
                    goodlistIm1.append(round(i, 6))
                else:
                    break
            imgs_in_range.append(round(max(goodlistIm1), 6))

        l = list(zip(reals_in_range,imgs_in_range))

        xs = np.array([x[0] for x in l])
        ys = np.array([x[1] for x in l])
        
        l = plt.plot(xs,ys,col[int(I)])
        lines = lines + [l]
        plt.plot(xs, -ys, col[int(I)])
        print('done m =', m)
    
    plt.title('Stability regions for IDC (order M-1) for varying M')
    st = lambda m: 'IDC M = ' + str(m)
    plt.legend([l[0] for l in lines], map(st,M))
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')
    plt.savefig('idc_stab_M.png')
    plt.show()


def RIDCtesteq(M,K,l, testeq):#Euler
    #(a,b)-endpoints, N-number of steps, p-order of method, K- No. intervals,  y0-I.C, F-function
     
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
            
        for l1 in range(1,M+1):   #correction
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


def ridc_stab(M_):
    # *Author: Sam Brossler
    f = ex.stab_eq
    col = ['y','g','c','m']
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    tests(3)
    for I, m_ in enumerate(M_):    
        goodlistRe = []
        goodlistIm = []
        #Finding the real values that for the stability interval
        for i in np.arange(0,2.005,0.005):
            M = RIDCtesteq(m_,10,-i,f)
            df = [abs(M[j]) for j in range(1,11)]
            r = max(df)
            if r<=1.:
                goodlistRe.append(round(-i,6))
        
        print('done reals')
        #print(goodlistRe)
        #finding the maximum imaginary values correcsponding to the real ones that satisfy the amplification factor
        for m in goodlistRe:
            goodlistIm1 = []
            for i in np.arange(0,1.005,0.005):
                M = RIDCtesteq(m_,10,complex(m,i),f)
                #print(M)
                dl = [abs(M[j]) for j in range(1,11)]
                #print(df)
                r = max(dl)
                #print(r)
                if r<=1.:
                    goodlistIm1.append(round(i,6))
            something = max(goodlistIm1)
            goodlistIm.append(round(something,11))
        print('done ims')

        l = list(zip(goodlistRe,goodlistIm))

        xs = np.array([x[0] for x in l])
        ys = np.array([x[1] for x in l])
        
        l = plt.plot(xs,ys,col[int(I)])
        lines = lines + [l]
        plt.plot(xs, -ys, col[int(I)])
        print('done m =', m_)
    plt.savefig('ridc_stab_M.png')
    plt.title('Stability regions for RIDC for varying M')
    st = lambda m: 'RIDC(M,10)_fe M = ' + str(m)
    plt.legend([l[0] for l in lines], map(st,M_))
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')
    plt.savefig('ridc_stab_M.png')
    plt.show()


if __name__ == '__main__':
    # plt.figure()
    # fe = tests(3)
    # rk4 = tests(5)
    # ab4 = tests(1)[0]
    # # plt.clabel(fe)
    # # plt.clabel(rk4)
    # # plt.clabel(ab4)

    # labels = ['FE', 'RK4', 'AB4']
    # fe.collections[0].set_label(labels[0])
    # rk4.collections[0].set_label(labels[1])
    # ab4.set_label(labels[2])

    # plt.legend(loc='upper right')


    idc_stab([2,3,5,6])

    
    # ridc_stab([2, 3, 4])

    #orders()
    