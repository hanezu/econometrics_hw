# coding: utf-8
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot
import matplotlib.pyplot as plt
from numpy import linalg as la

X = pd.read_excel(io='data2.xls',sheetname='paneldata_X')
Y = pd.read_excel(io='data2.xls',sheetname='paneldata_Y')
n = X.shape[0]
time_keys = ['t=1', 't=2']
X_vec = pd.concat([X['t=1'],X['t=2']],keys=time_keys)   # change matrix to vector
Y_vec = pd.concat([Y['t=1'],Y['t=2']],keys=time_keys)
X['delta'] = X['t=1']-X['t=2']
Y['delta'] = Y['t=1']-Y['t=2']
x_delta = X.delta
y_delta = Y.delta

#OLS
def Q1():
    # remember to add constant (unless don't need constant coeff when doing OLS
    model = sm.OLS(Y_vec,sm.add_constant(X_vec))
    results = model.fit()
    beta_OLS = X_vec.dot(Y_vec)/la.norm(X_vec)**2

    # the item 1 of params is beta_OLS, but why can't I call it by 'NaN'?
    print("result by OLS in statsmodels: %f, by direct calculation %f" % (float(results.params[1]), beta_OLS))
    fig,ax = plt.subplots()
    ax.plot(X_vec,Y_vec,'ro')
    X_extremes = np.array([max(X_vec), min(X_vec)])
    ax.plot(X_extremes,results.params[0] + results.params[1]*X_extremes,'b')
    plt.show()

#Fixed Effect beta
def Q2():
    # don't need constant here because of cancellation when doing transformation of fixed effect
    results = sm.OLS(y_delta,x_delta).fit()
    beta_FE = x_delta.dot(y_delta)/la.norm(x_delta)**2
    print("result by Fixed Effect in statsmodels: %f, by direct calculation %f" % (float(results.params), beta_FE))
    fig, ax = plt.subplots()
    ax.plot(x_delta,y_delta,'ro')
    X_extremes = np.array([max(x_delta), min(x_delta)])
    ax.plot(X_extremes,float(results.params)*X_extremes,'b')
    plt.show()



# another way to calc beta_FE
# trying to solve Q2 with the method on handout (instead of the book)
def calc_diff_of_mean(X):
    X['mean'] = (X['t=1'] + X['t=2']) / 2
    double_X_mean = pd.concat([X['mean'], X['mean']], keys=time_keys)
    X_with_mean = pd.concat([X_vec, double_X_mean], axis=1, keys=['data', 'mean'])
    assert isinstance(X_with_mean, pd.DataFrame)
    X_with_mean['diff'] = X_with_mean['data'] - X_with_mean['mean']
    return X_with_mean

def Q2_1():
    X_with_mean =calc_diff_of_mean(X)
    Y_with_mean =calc_diff_of_mean(Y)
    x_diff = X_with_mean['diff']
    y_diff = Y_with_mean['diff']
    results = sm.OLS(y_diff,x_diff).fit()
    beta_FE = x_diff.dot(y_diff)/la.norm(x_diff)**2
    print("result by Fixed Effect in statsmodels: %f, by direct calculation %f" % (float(results.params), beta_FE))
    fig, ax = plt.subplots()
    ax.plot(x_diff,y_diff,'ro')
    X_extremes = np.array([max(x_diff), min(x_diff)])
    ax.plot(X_extremes,float(results.params)*X_extremes,'b')
    plt.show()




def Q3():
    results = sm.OLS(y_delta,x_delta).fit()
    bunpo = la.norm(x_delta)**4
    beta_FE = float(results.params)
    u_hat = y_delta - beta_FE*x_delta
    bunsi = la.norm((u_hat**2).dot(x_delta**2))**2
    V_FE = n * bunsi/bunpo
    print('V_FE = %f' % (V_FE))

# like Q2_1
def Q3_1():
    X_with_mean =calc_diff_of_mean(X)
    Y_with_mean =calc_diff_of_mean(Y)
    x_diff = X_with_mean['diff']
    y_diff = Y_with_mean['diff']
    results = sm.OLS(y_diff,x_diff).fit()
    beta_FE = x_diff.dot(y_diff)/la.norm(x_diff)**2
    u_hat = y_diff - beta_FE*x_diff
    bunpo = la.norm(x_diff)**4
    bunsi = la.norm((u_hat**2).dot(x_diff**2))**2
    V_FE = n * bunsi/bunpo
    print('V_FE = %f' % (V_FE))


if __name__ == '__main__':
    Q3_1()