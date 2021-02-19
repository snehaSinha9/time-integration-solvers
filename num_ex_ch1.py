import numpy as np
from matplotlib import pyplot as plt

import test_examples as ex
from dc_experiments import DCs as dc
from serial_examples import Schemes as scheme


def analyse_scheme(dy_dt, y, a, b, ics, step_sizes, order_idc):
    max_global_err = []
    for i in range(len(step_sizes)):
        h = step_sizes[i]
        y_pred = scheme(v_field=dy_dt,
                        start=a, stop=b, h=h, init_conditions=ics)

        times = np.arange(a, b, h)
        exact = [y(t) for t in times]
        y_T = y(b)

        df = d_f({'times': times,
                  'euler': y_pred.euler()[1],
                  'rk4':  y_pred.rk4()[1],
                  'ab4': y_pred.ab4()[1],
                  'exact': exact})
        idc_err = []
        for order in order_idc:
            ts, df['idc_' + str(order)] = dc().idc_fe(a=a, b=b,
                                                      alpha=ics, N=int((b-a)/h), p=order, f=dy_dt)
            #ts, df['idc_' + str(order)] = IDC(a=a, b=b, N=int((b-a)/h), p =order, y0=ics)  #
            df['exact_'+str(order)] = [y(t) for t in ts]
            idc_err = idc_err + \
                [df['exact_'+str(order)].sub(df['idc_' +
                                                str(order)]).abs().max()]

        df['error_euler'] = df['exact'].sub(df['euler']).abs()
        df['error_rk4'] = df['exact'].sub(df['rk4']).abs()
        df['error_ab4'] = df['exact'].sub(df['ab4']).abs()
        max_global_err.append(idc_err+[df['error_euler'].max(),
                                       df['error_rk4'].max(),
                                       df['error_ab4'].max()])
    return max_global_err


def orders():
    def analyse_scheme(dy_dt, y, a, b, ics, step_sizes, order_idc):
        err = []
        y_t = y(b)
        def rel_err(y_p): return np.abs(y_t - y_p) / np.abs(y_t)

        for i in range(len(step_sizes)):
            h = step_sizes[i]
            y_pred = scheme(v_field=dy_dt,
                            start=a, stop=b, h=h, init_conditions=ics)

            err_fe = rel_err(y_pred.euler()[1][-1])
            err_rk4 = rel_err(y_pred.rk4()[1][-1])
            err_ab4 = rel_err(y_pred.ab4()[1][-1])

            h_errs = [err_fe, err_rk4, err_ab4]

            for order in order_idc:
                y_idc = dc().ridc_fe(a=a, b=b, alpha=ics, N=int(
                    (b-a)/h), p=order, K=order-1, f=dy_dt)[0]
                h_errs = h_errs + [rel_err(y_idc[-1])]

            #print(h, h_errs)

            err.append(h_errs)

        return np.array(err)

    h_sizes = [10**i for i in [-1, -2, -3]]  # , -3, -4]]

    start, stop = 0, 10

    N = [1/h for h in h_sizes]

    max_global_err = analyse_scheme(
        ex.func0, ex.y_exact0, start, stop, 1, h_sizes, [3, 6, 11])

    plt.plot(N, max_global_err)

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('log Relative Error')
    plt.xlabel('log Number of time-steps')
    plt.legend(['Forward Euler (order 1)', 'Runge-Kutta (order 4)', 'Adam Bashforth (order 4)',
                'IDC(3)-FE', 'IDC(6)-FE', 'IDC(11)-FE', ], loc='best', bbox_to_anchor=(1, 0.7))
    plt.show()

# 1 step
def stability_1step(R, axisbox=[-2, 2, -2, 2]):
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
    plt.contourf(X, Y, Rabs, levels, colors=[Sregion_color, 'w'])
    plt.contour(X, Y, Rabs, [1], colors='k')  # boundary

    # plot axes
    plt.plot([xa, xb], [0, 0], 'k')
    plt.plot([0, 0], [ya, yb], 'k')
    plt.axis(axisbox)
    plt.axis('scaled')  # so circles are circular
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')

# multi-step
def stability_mstep(rho, sigma, axisbox=[-2, 2, -2, 2]):
    theta = np.linspace(0, 2*np.pi, 1000)
    eitheta = np.exp(1j * theta)
    z = rho(eitheta) / sigma(eitheta)
    plt.plot(z.real, z.imag, 'r', linewidth=2)

    # plot axes
    xa, xb, ya, yb = axisbox
    plt.plot([xa, xb], [0, 0], 'k')
    plt.plot([0, 0], [ya, yb], 'k')
    plt.axis(axisbox)
    plt.axis('scaled')  # so circles are circular
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')


def tests(i):
    if (i == 1):
        # AB4
        def rho(z): return (z-1.) * z**3
        def sigma(z): return (((55*z - 59)*z + 37.)*z - 9) / 24.
        stability_mstep(rho, sigma)
    elif (i == 2):
        # BE
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1./(1.-z)
        stability_1step(R, [-2.5, 2.5, -2, 2])
    elif (i == 3):
        # FE
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1./(1.+z)
        stability_1step(R, [-2.5, 2.5, -2, 2])
    elif (i == 4):
        # CN
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return (1. + z/2.)/(1.-z/2.)
        stability_1step(R, [-2.5, 2.5, -2, 2])
    elif (i == 5):
        # RK4
        np.seterr(all='ignore')  # ignore divide by zero errors
        def R(z): return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
        stability_1step(R, [-5, 5, -5, 5])
    elif (i == 6):
        # Implicit trapozodial
        def rho(z): return z-1.
        def sigma(z): return z/2. + 0.5
        stability_mstep(rho, sigma, [-2, 2, -2, 2])
    elif (i == 7):
        # Implicit trapozodial
        def rho(z): return z-1.
        def sigma(z): return 1
        stability_mstep(rho, sigma, [-2, 2, -2, 2])

# !idc-stab, does not work

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
            def func_lambda(t, y): return stab_func(z, t, y)
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

        # finding the maximum imaginary values correcsponding to 
        # the real ones that satisfy the amplification factor
        for r in reals_in_range:
            goodlistIm1 = []
            for i in np.arange(0, 3.01, 0.01):
                ims_p = dc().idc_stability(complex(r, i), m, f)
                if abs(ims_p[-1]) <= 1.:
                    goodlistIm1.append(round(i, 6))
           
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


def ridc_stab(M):
    # *Author: Sam Brossler
    f = ex.stab_eq
    col = ['y','g','c','m']
    lines = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for I, m in enumerate(M):    
        reals_in_range = []
        imgs_in_range = []
        # Finding the real values that for the stability interval
        for i in np.arange(0, 3.01, 0.01):
            reals_p = dc().ridc_fe_stab(4,10,-i,f)
            if abs(reals_p[-1]) <= 1.:
                reals_in_range.append(round(-i, 6))

        # finding the maximum imaginary values correcsponding to 
        # the real ones that satisfy the amplification factor
        for r in reals_in_range:
            goodlistIm1 = []
            for i in np.arange(0, 3.01, 0.01):
                ims_p = dc().ridc_fe_stab(4,10,-i,f)
                if abs(ims_p[-1]) <= 1.:
                    goodlistIm1.append(round(i, 6))
           
            imgs_in_range.append(round(max(goodlistIm1), 6))

        l = list(zip(reals_in_range,imgs_in_range))

        xs = np.array([x[0] for x in l])
        ys = np.array([x[1] for x in l])
        
        l = plt.plot(xs,ys,col[int(I)])
        lines = lines + [l]
        plt.plot(xs, -ys, col[int(I)])
        print('done m =', m)
    plt.savefig('ridc_stab_M.png')
    plt.title('Stability regions for RIDC for varying M')
    st = lambda m: 'RIDC(4,10)_fe M = ' + str(m)
    plt.legend([l[0] for l in lines], map(st,M))
    plt.xlabel(r'$Re(\lambda \Delta t)$')
    plt.ylabel(r'$Im(\lambda \Delta t)$')
    plt.savefig('ridc_stab_M.png')
    plt.show()


if __name__ == '__main__':
    # tests(2)
    #idc_stab([2, 3, 5, 6])
    ridc_stab([2, 3, 5, 6])
    