# coding: utf-8
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot
import matplotlib.pyplot as plt
X = pd.read_excel(io='data2.xls',sheetname='paneldata_X')
Y = pd.read_excel(io='data2.xls',sheetname='paneldata_Y')
pd.concatenate([Y['t=1'],Y['t=2']])
pd.concat([Y['t=1'],Y['t=2']])
Y_vec = pd.concat([Y['t=1'],Y['t=2']])
X_vec = pd.concat([X['t=1'],X['t=2']])
model = sm.OLS(vec_Y,vec_X)
model = sm.OLS(Y_vec,X_vec)
results = model.fit()
results.params
plt.plot(X_vec,Y_vec)
plt.plot(X_vec,Y_vec,'ro')
plt.plot(X_vec['NaN'],Y_vec,'ro')
plt.plot(X_vec.ix[:,2],Y_vec,'ro')
plt.plot(X_vec.ix[:,1],Y_vec,'ro')
results.params
X_vec
X
X = pd.read_excel(io='data2.xls',sheetname='paneldata_X')
Y
X
X['delta'] = X['t=1']-X['t=2']
X
Y['delta'] = Y['t=1']-Y['t=2']
x_delta = X.delta
y_delta = Y.delta
x_delta
results = sm.OLS(y_delta,sm.add_constant(x_delta)).fit()
results.params
x_delta
y_delta
results = sm.OLS(y_delta,x_delta).fit()
results.params
from numpy import linalg as la
la.norm(x_delta)
la.norm(x_delta)
y_delta * x_delta
import numpy as np
np.multiply(x_delta,y_delta)
x_delta.dot(y_delta)
x_delta.dot(y_delta)/la.norm(x_delta)**2
bunpo = la.norm(x_delta)**4
bunpo
result.params
results.params
bfe = results.params
u_hat = y_delta - bfe*x_delta
u_hat
bfe = results.params
bfe
bfe = eval(results.params)
bfe = double(results.params)
bfe = float(results.params)
bfe
u_hat = y_delta - bfe*x_delta
u_hat
bunsi = la.norm((u_hat**2).dot(x_delta**2))**2
bunsi
bunbo
bunpo
u
u_hat
plt.plot(x_delta,y_delta)
plt.show()
plt.plot(x_delta,y_delta,'.')
plt.show()
X
x_2 = pd.concat([X['t=1'],X['t=2']])
x_2
X['mean'] = (X['t=1']+X['t=2'])/2
X
x_2['mean']=pd.concat([X['mean'],X['mean']])
x_2
x_2.size
x_2.shape
x_2['mean']
del x_2['mean']
x_2
X_2 = pd.concat([x_2,pd.concat([X['mean'],X['mean']])])
X_2
X_2 = pd.concat([x_2,pd.concat([X['mean'],X['mean']])], axis='1')
X_2 = pd.concat([x_2,pd.concat([X['mean'],X['mean']])], axis=1)
X_2
X_2.rename(columns=['X','X_mean'])
X_2.columns=['X','X_mean']
X_2
X_2['diff'] = X_2['X'] - X_2.X_mean
X_2['diff']
