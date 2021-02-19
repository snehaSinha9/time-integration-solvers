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


def test1():
    N = 320
    start, stop = 0, 1
    h = (stop-start)/N
    ts = np.arange(start, stop, h)

    dy_dt = ex.func0
    y = ex.y_exact0
    y0 = 1
    #yN = y(stop)

    serial = scheme(v_field=dy_dt,
                    start=start,
                    stop=stop,
                    h=h,
                    init_conditions=y0)

    _, fe, fe_time = serial.euler()
    _, rk4, rk4_time = serial.rk4()
    _, ab4, ab4_time = serial.ab4()

    true = np.array([y(t) for t in ts])
    fe_err = np.mean(np.abs(true-fe))
    rk4_err = np.mean(np.abs(true - rk4))
    ab4_err = np.mean(np.abs(true - ab4))

    idc_fe2, time_idc_fe2 = dc().idc_fe(a=start, b=stop, alpha=y0, N=N, p=3, f=dy_dt)
    idc_fe3, time_idc_fe3 = dc().idc_fe(a=start, b=stop, alpha=y0, N=N, p=5, f=dy_dt)
    idc_fe6, time_idc_fe6 = dc().idc_fe(a=start, b=stop, alpha=y0, N=N, p=6, f=dy_dt)
    idc_fe2_err = np.mean(np.abs(true-idc_fe2))
    idc_fe3_err = np.mean(np.abs(true-idc_fe3))
    idc_fe6_err = np.mean(np.abs(true-idc_fe6))

    ridc_fe3, time_ridc_fe3 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=N, f=dy_dt)
    ridc_fe4, time_ridc_fe4 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/2), f=dy_dt)
    ridc_fe5, time_ridc_fe5 = dc().ridc_fe(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/4), f=dy_dt)
    ridc_fe3_err = np.mean(np.abs(true-ridc_fe3[1:]))
    ridc_fe4_err = np.mean(np.abs(true-ridc_fe4[1:]))
    ridc_fe5_err = np.mean(np.abs(true-ridc_fe5[1:]))

    ridc_rk6, time_ridc_rk6 = dc().ridc_rk4(
        a=start, b=stop, alpha=y0, N=N, p=4, K=N, f=dy_dt)
    ridc_rk7, time_ridc_rk7 = dc().ridc_rk4(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/2), f=dy_dt)
    ridc_rk8, time_ridc_rk8 = dc().ridc_rk4(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/4), f=dy_dt)
    ridc_rk6_err = np.mean(np.abs(true-ridc_rk6[1:]))
    ridc_rk7_err = np.mean(np.abs(true-ridc_rk7[1:]))
    ridc_rk8_err = np.mean(np.abs(true-ridc_rk8[1:]))

    ridc_ab9, time_ridc_ab9 = dc().ridc_ab2(
        a=start, b=stop, alpha=y0, N=N, p=4, K=N, f=dy_dt)
    ridc_ab10, time_ridc_rk10 = dc().ridc_ab2(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/2), f=dy_dt)
    ridc_ab11, time_ridc_rk11 = dc().ridc_ab2(
        a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/4), f=dy_dt)
    ridc_ab9_err = np.mean(np.abs(true-ridc_ab9[1:]))
    ridc_ab10_err = np.mean(np.abs(true-ridc_ab10[1:]))
    ridc_ab11_err = np.mean(np.abs(true-ridc_ab11[1:]))

    ridc_abM, time_ridc_abM = dc().ridc_abM(
        T=stop, y0=[y0], N=N, M=4, approach=0, f=dy_dt)
    ridc_abM_err = np.mean(np.abs(true-ridc_abM[1:]))

    # (a,b)-endpoints, N-number of steps, p-order of method, K- No. intervals,  y0-I.C, F-function
    ridc_sam, time_ridc_sam = dc().sam_ridc_fe(a=start, b=stop, alpha=y0, N=N, p=4, K=int(N/4), f=dy_dt)
    ridc_sam_err = np.mean(np.abs(true-ridc_sam[1:]))

    ind = ['FE', 'RK4', 'AB4',
           'IDC3-FE', 'IDC5-FE', 'IDC6-FE',
           'RIDC(4,N)-FE', 'RIDC(4,N/2)-FE', 'RIDC(4,N/4)-FE',
           'RIDC(4,N)-RK4', 'RIDC(4,N/2)-RK4', 'RIDC(4,N/4)-RK4',
           'RIDC(4,N)-AB2', 'RIDC(4,N/2)-AB2', 'RIDC(4,N/4)-AB2', 'hossein RIDC', 'sam RIDC-FE']

    def change_t(x): return "%.3e" % x
    def change_e(x): return "%.4e" % x

    times = list(map(change_t, [fe_time, rk4_time, ab4_time,
                                time_idc_fe2, time_idc_fe3, time_idc_fe6,
                                time_ridc_fe3, time_ridc_fe4, time_ridc_fe5,
                                time_ridc_rk6, time_ridc_rk7, time_ridc_rk8,
                                time_ridc_ab9, time_ridc_rk10, time_ridc_rk11,
                                time_ridc_abM, time_ridc_sam]))

    errors = list(map(change_e, [fe_err, rk4_err, ab4_err,
                                 idc_fe2_err, idc_fe3_err, idc_fe6_err,
                                 ridc_fe3_err, ridc_fe4_err, ridc_fe5_err,
                                 ridc_rk6_err, ridc_rk7_err, ridc_rk8_err,
                                 ridc_ab9_err, ridc_ab10_err, ridc_ab11_err,
                                 ridc_abM_err, ridc_sam_err]))

    n = [N]*len(times)
    k = [np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan,
         320, 160, 80,
         320, 160, 80,
         320, 160, 80, np.nan, np.nan]

    data = {'N': n, 'K': k, 'Time': times, 'Average Error': errors}

    # Creates pandas DataFrame.
    df_ = df(data, index=ind)

    # print the data
    print(df_)
    # print(df_.to_latex())


def test2(method):
    dy_dt = ex.func1
    y = ex.y_exact1

    rangeN = [50*(i**2) for i in [2,4,6,8,10] ]
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

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))
    for i, eK in enumerate(errorsK0):
        ax0.plot(rangeN, eK, markersize=2)
        print('1: Slope in first grid K ' + str(ks[i]) +' is '+ str(slopes(rangeN[:2], eK[:2])))
    ax0.grid()
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.legend(list(map(lambda x: "K = " + str(x), ks)))
    ax0.set_xlabel('Number of intervals')
    ax0.set_ylabel(r'Maximum absolute error')

    for i, eK in enumerate(errorsK1):
        ax1.plot(rangeN, eK, markersize=2)
        print('2: Slope in first grid K ' + str(ks[i]) +' is '+ str(slopes(rangeN[:2], eK[:2])))
    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(list(map(lambda x: "K = " + str(x), ks)))
    ax1.set_xlabel('Number of intervals')
    ax1.set_ylabel(r'Maximum absolute error')
    plt.show()

    errorsK0, errorsK1


def test3(method):
    dy_dt = ex.func1
    y = ex.y_exact1

    errorsM0, errorsM1 = [], []
    M = [1,2,3,4,5,6]
    rangeN = [50*(i**2) for i in [2,4,6,8,10] ]

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

    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))

    for i,eM in enumerate(errorsM0):
        ax0.plot(rangeN, eM, markersize=2)
        print('1: Slope in first grid M ' + str(M[i]) +' is ', slopes(rangeN[:2], eM[:2]))
    ax0.grid()
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.legend(list(map(lambda x: "M = " + str(x), M)))
    ax0.set_xlabel('Number of intervals')
    ax0.set_ylabel(r'Maximum absolute error')

    for i, eM in enumerate(errorsM1):
        ax1.plot(rangeN, eM, markersize=2)
        print('2: Slope in first grid M ' + str(M[i]) +' is ', slopes(rangeN[:2], eM[:2]))
    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(list(map(lambda x: "M = " + str(x), M)))
    ax1.set_xlabel('Number of intervals')
    ax1.set_ylabel(r'Maximum absolute error')
    plt.show()


def test4():
    dy_dt = ex.func2
    y = ex.y_exact2

    
    rangeN = [50*(i**2) for i in [2,4,6,8,10] ]
    errorsM0, errorsM1 = [], []
    approach = [0,1,2]

    for a in approach:
        errors0, errors1 = [], []
        for n in rangeN:
            t = np.linspace(0, 1, n+1)
            y_true = y(t)
            y_pred = np.array(dc().ridc_abM(T=1, y0=(1, 1), N=n, M = 4, approach = a, f=dy_dt)[0])

            error = max(np.absolute(
                y_true[0] - y_pred[:, 0])), max(np.absolute(y_true[1] - y_pred[:, 1]))
            errors0, errors1 = errors0 + [error[0]], errors1 + [error[1]]
        errorsM0 = errorsM0 + [errors0]
        errorsM1 = errorsM1 + [errors1]

    _, (ax0, ax1) = plt.subplots(ncols=2, figsize=(18, 6))

    for i, eM in enumerate(errorsM0):
        ax0.plot(rangeN, eM, markersize=2)
        print('1: Slope in first grid approach ' + str(approach[i]) +' is ', slopes(rangeN[:2], eM[:2]))
    ax0.grid()
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.legend(list(map(lambda x: "approach = " + str(x), approach)))
    ax0.set_xlabel('Number of intervals')
    ax0.set_ylabel(r'Maximum absolute error')
    #ax0.title('First dimension')

    for i, eM in enumerate(errorsM1):
        ax1.plot(rangeN, eM, markersize=2)
        print('2: Slope in first grid approach ' + str(approach[i]) +' is ', slopes(rangeN[:2], eM[:2]))
    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(list(map(lambda x: "approach = " + str(x), approach)))
    ax1.set_xlabel('Number of intervals')
    ax1.set_ylabel(r'Maximum absolute error')
    #ax1.set_title('Second dimension')
    plt.show()


def test5():
    T1 = 10
    M = 4
    y0 = [1,0]
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

        err0[i] = (yExact[0]-yy_0[-1,0])**2 + (yExact[1]-yy_0[-1,1])**2
        err1[i] = (yExact[0]-yy_1[-1, 0])**2 + (yExact[1]-yy_1[-1,1])**2
        err2[i] = (yExact[0]-yy_2[-1, 0])**2 + (yExact[1]-yy_2[-1,1])**2

    fig, ax = plt.subplots()
    ax.plot(rangeN, err0, label='approach 0')
    ax.plot(rangeN, err1, label='approach 1')
    ax.plot(rangeN, err2, label='approach 2')
    ax.plot(rangeN, rangeNpower, label='e = N^{-9}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')  # Add an x-label to the axes.
    ax.set_ylabel('squared amplitude error')
    ax.set_title("Errors using different methods")
    ax.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    # table, test time and error
    #test1()

    # figure, test K
    #test2(dc().ridc_rk4)

    # figure test M
    test3(dc().ridc_fe)

    #test4() 
    #test5()