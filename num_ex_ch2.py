from pandas import DataFrame as df
import numpy as np
from dc_experiments import DCs as dc
from serial_examples import Schemes as scheme
import test_examples as ex
from time import time
from functools import wraps
from matplotlib import pyplot as plt
from scipy.stats import linregress


def slopes(xs, ys):
    return linregress(np.log(np.array(xs)), np.log(np.array(ys)))[0]

# ! change to do the time thing
def test1():
    N = 320000
    start, stop = 0, 1
    h = (stop-start)/N
    #ts = np.arange(start, stop, h)

    dy_dt = ex.func0
    y = ex.y_exact0
    y0 = 1
    #yN = y(stop)

    sigma = lambda m, k : 1 + m**2/k

    _, _, fe_time = scheme(v_field=dy_dt,
                        start=start,
                        stop=stop,
                        h=h,
                        init_conditions=y0).euler()
    print(fe_time)
    sigmas = [sigma(3, k) for k in [N, N/2, N/4]]
    theoretical_time = [fe_time*s for s in sigmas]
    print( sigmas, theoretical_time)   
    _, time_ridc_fe3 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=N, f=dy_dt)
    _, time_ridc_fe4 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/2), f=dy_dt)
    _, time_ridc_fe5 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/4), f=dy_dt)

    
    actual_time = [time_ridc_fe3, time_ridc_fe4, time_ridc_fe5]
    print(actual_time)


def test2(method):
    dy_dt = ex.func1
    y = ex.y_exact1

    rangeN = [50*(i**2) for i in [2, 4, 6, 8, 10]]
    ks = [5, 10, 20, 25, 50]
    errorsK0 = []
    errorsK1 = []

    for k in ks:
        errors0, errors1 = [], []
        for n in rangeN:
            t = np.linspace(0, 1, n+1)
            y_true = y(t)
            y_pred = np.array(method(a=0, b=1, alpha=(1, 1),
                                     N=n, p=5, K=k, f=dy_dt)[0])

            error = max(np.absolute(
                y_true[0] - y_pred[:, 0])), max(np.absolute(y_true[1] - y_pred[:, 1]))
            errors0, errors1 = errors0 + [error[0]], errors1 + [error[1]]

        errorsK0 = errorsK0 + [errors0]
        errorsK1 = errorsK1 + [errors1]

    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))
    # for i, eK in enumerate(errorsK0):
    #     ax0.plot(rangeN, eK, markersize=2)
    #     print('1: Slope in first grid K ' +
    #           str(ks[i]) + ' is ' + str(slopes(rangeN[:2], eK[:2])))
    # ax0.grid()
    # ax0.set_xscale('log')
    # ax0.set_yscale('log')
    # ax0.legend(list(map(lambda x: "K = " + str(x), ks)))
    # ax0.set_xlabel('Number of time steps')
    # ax0.set_ylabel(r'Maximum absolute error')

    # for i, eK in enumerate(errorsK1):
    #     ax1.plot(rangeN, eK, markersize=2)
    #     print('2: Slope in first grid K ' +
    #           str(ks[i]) + ' is ' + str(slopes(rangeN[:2], eK[:2])))
    # ax1.grid()
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.legend(list(map(lambda x: "K = " + str(x), ks)))
    # ax1.set_xlabel('Number of time steps')
    # ax1.set_ylabel(r'Maximum absolute error')
    # plt.show()

    # errorsK0, errorsK1
    
    # just first dimension
    plt.clf()
    plt.figure(figsize=(16,8))
    for i, eK in enumerate(errorsK0):
        plt.plot(rangeN, eK, markersize=2)
        print('1: Slope in first grid K ' +
              str(ks[i]) + ' is ' + str(slopes(rangeN[:2], eK[:2])))
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(list(map(lambda x: "K = " + str(x), ks)))
    plt.xlabel('Number of time steps')
    plt.ylabel(r'Maximum absolute error')
    plt.show()


def test3(method):
    dy_dt = ex.func1
    y = ex.y_exact1

    errorsM0, errorsM1 = [], []
    M = [1, 2, 3, 4, 5, 6]
    rangeN = [50*(i**2) for i in [2, 4, 6, 8, 10]]

    for m in M:
        errors0, errors1 = [], []
        for n in rangeN:
            t = np.linspace(0, 1, n+1)
            y_true = y(t)
            y_pred = np.array(method(a=0, b=1, alpha=(
                1, 1), N=n, p=m+1, K=10, f=dy_dt)[0])

            error = max(np.absolute(
                y_true[0] - y_pred[:, 0])), max(np.absolute(y_true[1] - y_pred[:, 1]))
            errors0, errors1 = errors0 + [error[0]], errors1 + [error[1]]
        errorsM0 = errorsM0 + [errors0]
        errorsM1 = errorsM1 + [errors1]

    # fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))

    # for i, eM in enumerate(errorsM0):
    #     ax0.plot(rangeN, eM, markersize=2)
    #     print('1: Slope in first grid M ' +
    #           str(M[i]) + ' is ', slopes(rangeN[:2], eM[:2]))
    # ax0.grid()
    # ax0.set_xscale('log')
    # ax0.set_yscale('log')
    # ax0.legend(list(map(lambda x: "M = " + str(x), M)))
    # ax0.set_xlabel('Number of time steps')
    # ax0.set_ylabel(r'Maximum absolute error')

    # for i, eM in enumerate(errorsM1):
    #     ax1.plot(rangeN, eM, markersize=2)
    #     print('2: Slope in first grid M ' +
    #           str(M[i]) + ' is ', slopes(rangeN[:2], eM[:2]))
    # ax1.grid()
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax1.legend(list(map(lambda x: "M = " + str(x), M)))
    # ax1.set_xlabel('Number of time steps')
    # ax1.set_ylabel(r'Maximum absolute error')
    # plt.show()
    plt.clf()
    plt.figure(figsize=(16,8))
    for i, eM in enumerate(errorsM0):
        plt.plot(rangeN, eM, markersize=2)
        print('1: Slope in first grid M ' +
              str(M[i]) + ' is ', slopes(rangeN[:2], eM[:2]))
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(list(map(lambda x: "M = " + str(x), M)))
    plt.xlabel('Number of time steps')
    plt.ylabel(r'Maximum absolute error')
    plt.show()


def test4():
    dy_dt = ex.func2
    y = ex.y_exact2

    rangeN = [50*(i**2) for i in [2, 4, 6, 8, 10]]
    errorsM0, errorsM1 = [], []
    approach = [0, 1, 2]

    for a in approach:
        errors0, errors1 = [], []
        for n in rangeN:
            t = np.linspace(0, 1, n+1)
            y_true = y(t)
            y_pred = np.array(dc().ridc_abM(
                T=1, y0=(1, 1), N=n, M=4, approach=a, f=dy_dt)[0])

            error = max(np.absolute(
                y_true[0] - y_pred[:, 0])), max(np.absolute(y_true[1] - y_pred[:, 1]))
            errors0, errors1 = errors0 + [error[0]], errors1 + [error[1]]
        errorsM0 = errorsM0 + [errors0]
        errorsM1 = errorsM1 + [errors1]

    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))

    for i, eM in enumerate(errorsM0):
        ax0.plot(rangeN, eM, markersize=2)
        print('1: Slope in first grid approach ' +
              str(approach[i]) + ' is ', slopes(rangeN[:2], eM[:2]))
    ax0.grid()
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.legend(list(map(lambda x: "approach = " + str(x), approach)))
    ax0.set_xlabel('Number of time steps')
    ax0.set_ylabel(r'Maximum absolute error')
    # ax0.title('First dimension')

    for i, eM in enumerate(errorsM1):
        ax1.plot(rangeN, eM, markersize=2)
        print('2: Slope in first grid approach ' +
              str(approach[i]) + ' is ', slopes(rangeN[:2], eM[:2]))
    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(list(map(lambda x: "approach = " + str(x), approach)))
    ax1.set_xlabel('Number of timesteps')
    ax1.set_ylabel(r'Maximum absolute error')
    # ax1.set_title('Second dimension')
    plt.show()


def test5():
    T1 = 10
    M = 4
    y0 = [1, 0]
    dy_dt = ex.func2
    y_true = ex.y_exact2
    # Y_exact = (1 + np.linspace(0, T1, N)**2)**2
    yExact = y_true([T1])
    rangeN = [int(10**n) for n in np.arange(1.8, 3.8, 0.2)]
    rangeNpower = [(10**12)*int(10**n)**(-9) for n in np.arange(1.8, 3.8, 0.2)]

    err0 = np.empty(np.shape(rangeN))
    err1 = np.empty(np.shape(rangeN))
    err2 = np.empty(np.shape(rangeN))
    for i, NN in enumerate(rangeN):
        yy_0, _ = dc().ridc_abM(T1, y0, NN-1, M, 0, dy_dt)
        yy_1, _ = dc().ridc_abM(T1, y0, NN-1, M, 1, dy_dt)
        yy_2, _ = dc().ridc_abM(T1, y0, NN-1, M, 2, dy_dt)

        err0[i] = (yExact[0]-yy_0[-1, 0])**2 + (yExact[1]-yy_0[-1, 1])**2
        err1[i] = (yExact[0]-yy_1[-1, 0])**2 + (yExact[1]-yy_1[-1, 1])**2
        err2[i] = (yExact[0]-yy_2[-1, 0])**2 + (yExact[1]-yy_2[-1, 1])**2

    fig, ax = plt.subplots()
    ax.plot(rangeN, err0, label='approach 0')
    ax.plot(rangeN, err1, label='approach 1')
    ax.plot(rangeN, err2, label='approach 2')
    ax.plot(rangeN, rangeNpower, label='e = N^{-9}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')  # Add an x-label to the axes.
    ax.set_ylabel('Squared amplitude error')
    ax.set_title("Errors using different methods")
    ax.legend(loc='lower left')
    plt.show()


def test6():
    dy_dt = ex.func0
    y = ex.y_exact0

    T1 = 100
    M = 4
    y0 = np.array([1])
    yExact = y(T1)
    rangeN = [int(10**n) for n in np.arange(1.8, 3.8, 0.2)]
    rangeNpower = [(3*10**13)*int(10**n)**(-8)
                    for n in np.arange(1.8, 3.8, 0.2)]
    rangeNpower2 = [(3*10**13)*int(10**n)**(-9)
                     for n in np.arange(1.8, 3.8, 0.2)]

    err0 = np.empty(np.shape(rangeN))
    err1 = np.empty(np.shape(rangeN))
    # err2 = np.empty(np.shape(rangeN))
    err3 = np.empty(np.shape(rangeN))

    for i, NN in enumerate(rangeN):
        yy_0, _ = dc().ridc_abM(T=T1, y0=[1], N=NN-1, M=M, approach=0, f=dy_dt)
        yy_1, _ = dc().ridc_abM(T=T1, y0=[1], N=NN-1, M=M, approach=1, f=dy_dt)
        # yy_2 = dc().ridc_hosseinAB(func=dy_dt, T=T1, y0=y0, N=NN-1, M=M)
        yy_3, _ = dc().ridc_abM(T=T1, y0=[1], N=NN-1, M=M, approach=2, f=dy_dt)

        err0[i] = abs((yExact-yy_0[-1])/yExact)
        err1[i] = abs((yExact-yy_1[-1])/yExact)
        # err2[i] = abs((yExact-yy_2[-1][-1])/yExact)
        err3[i] = abs((yExact-yy_3[-1])/yExact)

    fig, ax = plt.subplots()
    ax.plot(rangeN, err0, label='apprach 0')
    ax.plot(rangeN, err1, label='approach 1')
    ax.plot(rangeN, err3, label='approach 2')
    # ax.plot(rangeN, err2, label='Pred AB4, Corr AB3')
    ax.plot(rangeN, rangeNpower, '--', label='e = N^{-8}')
    ax.plot(rangeN, rangeNpower2, '--', label='e = N^{-9}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')  # Add an x-label to the axes.
    ax.set_ylabel('$|y - y_{exact}|$')
    ax.set_title("Errors using different methods")
    ax.legend(loc='lower left')
    plt.show()


def test7():
    dy_dt = ex.func0
    y = ex.y_exact0

    T1 = 100
    y0 = np.array([1])
    yExact = y(T1)
    ranges = np.arange(2.0, 3.6, 0.2)
    rangeN = [int(10**n) for n in ranges]
    rangeNpower = [(6*10**7)*int(10**n)**(-4) for n in ranges]
    rangeNpower2 = [(36*10**8)*int(10**n)**(-9) for n in ranges]
    rangeNpower3 = [(6*10**18)*int(10**n)**(-12) for n in ranges]

    err0 = np.empty(np.shape(rangeN))
    err1 = np.empty(np.shape(rangeN))
    err2 = np.empty(np.shape(rangeN))
    err3 = np.empty(np.shape(rangeN))

    for i, NN in enumerate(rangeN):
        # rk8
        yy_0 = dc().ridc_abM2(T=T1, y0=[1], N=NN-1, M=3, approach=0, f=dy_dt)
        # rk4
        yy_1 = dc().ridc_abM(T=T1, y0=[1], N=NN-1, M=5, approach=0, f=dy_dt)
        yy_2 = dc().ridc_hosseinAB(func=dy_dt, T=T1, y0=y0, N=NN-1, M=4)
        yy_3 = dc().ridc_hossein_test1(func=dy_dt, T=T1, y0=y0, N=NN-1, M=4)

        err0[i] = abs((yExact-yy_0[-1])/yExact)
        err1[i] = abs((yExact-yy_1[-1])/yExact)
        err2[i] = abs((yExact-yy_2[-1][-1])/yExact)
        err3[i] = abs((yExact-yy_3[-1][-1])/yExact)

    fig, ax = plt.subplots()

    ax.plot(rangeN, err1, label='M=5, Start-up RK4, Prediction AB5, Correctors AB4')
    ax.plot(rangeN, err3, label='M=4, Start-up RK8, Prediction AB4, Correctors AB3')
    ax.plot(rangeN, err2, label='M=4, Start-up RK4, Prediction AB4, Correctors AB3')
    ax.plot(rangeN, err0, label='M=3, Start-up RK8, Prediction AB3, Correctors AB2')

    ax.plot(rangeN, rangeNpower, '--', label='e = N^{-8}')
    ax.plot(rangeN, rangeNpower2, '--', label='e = N^{-9}')
    ax.plot(rangeN, rangeNpower3, '--', label='e = N^{-12}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')  # Add an x-label to the axes.
    ax.set_ylabel('$|y - y_{exact}|$')
    ax.set_title("Errors using different methods")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()

def test8(method):

    dy_dt = ex.func1
    y = ex.y_exact1
    a, b = 0,1
    rangeN = [50*(i**2) for i in [2, 4, 6, 8]]
    ks = [5, 10, 25, 50]
    M=4
    T=1

    for k in ks:
        for N in rangeN:
            plt.clf()
            dt = float(T)/N
            dx = (b-a)/M

            y_preds = np.array(method(a=a, b=b, alpha=(1, 1),
                                        N=N, p=M+1, K=k, f=dy_dt)[0])

            xs = np.arange(a, b + dx, dx)
            ts = np.arange(0, T + dt, dt)
            ns = [int(N/20), int(N/10), int(N/4), int(N/2), int(N)]
            for n in ns:
                y_true_t = np.reshape(y([ts[n]]), (2,))
                y_pred_t = y_preds[n]
                err = np.linalg.norm(y_pred_t - y_true_t, ord=np.inf)
                plt.plot(xs, err, label= 't='+ str(ts[n]))
                
            plt.legend(loc='best')
            plt.grid(True)
            plt.xlabel('xs')
            plt.ylabel('log|err|')
            plt.yscale('log')
            plt.title('RIDC with RK4 predictors \n with N = '+ str(N) +',  M=4, K='+ str(k))
            plt.savefig('/home/rhea/Documents/project/dissertation-files/proj_imgs/ridc_rk4_'+str(N) + '_' +str(k)+'.png')
            plt.show()

if __name__ == '__main__':
    # table, test time and error
    # test1()

    # figure, test K
    # test2(dc().ridc_rk4)

    # figure test M
    # test3(dc().ridc_rk4)

    # test4()
    # test5()

    # testing 5 but better
    # test6()

    # testing approaches
    test7()

    # testing 2, changes in K as time progresses
    # test8(dc().ridc_rk4)



