import numpy as np

def func0(t, y):
    if isinstance(y, list):
        y = y[0]
    return 4*t*y**(0.5)

def y_exact0(t):
    return (1 + t**2)**2


def func1(t, y):
    y1 = y[0]
    y2 = y[1]
    y1_p = t*y2 + y1  # -y2 + y1*(1-y1**2-y2**2)
    y2_p = -t*y1 + y2  # y1 + 3*y2*(1-y1**2-y2**2)
    return np.array([y1_p, y2_p])

def y_exact1(ts):
    def y1(t): return np.exp(t) * (np.cos(0.5*t**2)+np.sin(0.5*t**2))
    def y2(t): return np.exp(t) * (np.cos(0.5*t**2)-np.sin(0.5*t**2))
    return np.array([[y1(t) for t in ts], [y2(t) for t in ts]])


#y_0 =(1,0) , t in 0, 10
def func2(t,y):
    y_1 = y[0]
    y_2 = y[1]
    y1_p = -y_2 + y_1*(1-y_1**2 - y_2**2)
    y2_p = y_1 + 3*y_2*(1-y_1**2 - y_2**2)
    return np.array([y1_p, y2_p])

def y_exact2(ts):
    return np.array([[np.cos(t) for t in ts], [np.sin(t) for t in ts]])

def stab_eq(l,y):
    return l*y