#!/usr/bin/env python3

import math
import numpy as np
from sklearn import linear_model, metrics
from sklearn.utils import check_arrays

data_all = []
data = []
print("Carregando base...")
with open("Polinômio.txt") as f:
  for line in f:
    l = line.split("  ")
    l = list(map(lambda x:x.strip(), l))
    data.append(l);

data = np.array(data[1:]).astype(np.float64)
print("Carregado: %d amostras" % len(data))

half_len = len(data)/2
print("Particionando: %d amostras" % half_len)

X = np.array(data[:,0]).astype(np.float64);
Y = np.array(data[:,1]).astype(np.float64);

X_part1 = X[0:half_len];
Y_part1 = Y[0:half_len];

X_part2 = X[half_len:];
Y_part2 = Y[half_len:];

X_array = list(map(lambda x:[x], X))
X_part1_array = list(map(lambda x:[x], X_part1))
X_part2_array = list(map(lambda x:[x], X_part2))

model_part = linear_model.LinearRegression()
model_part.fit(X_part1_array, Y_part1)
print(model_part.coef_, model_part.intercept_)

def mean_absolute_percentage_error(y_true, y_pred): 
  y_true, y_pred = check_arrays(y_true, y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

Y_pred_part2 = model_part.predict(X_part2_array)
MSE = metrics.mean_squared_error(Y_part2, Y_pred_part2)
RMSE = math.sqrt(MSE)
print("PART2: MSE: %0.4f" % MSE);
print("PART2: RMSE: %0.4f" % RMSE);
MAPE = mean_absolute_percentage_error(Y_part2, Y_pred_part2);
print("PART2: MAPE: %0.4f" % MAPE);



model = linear_model.LinearRegression()
model.fit(X_array, Y)
print(model.coef_, model.intercept_)

Y_pred = model.predict(X_array)
MSE = metrics.mean_squared_error(Y, Y_pred)
RMSE = math.sqrt(MSE)
MAPE = mean_absolute_percentage_error(Y, Y_pred);
print("TOTAL: MSE: %0.4f" % MSE);
print("TOTAL: RMSE: %0.4f" % RMSE);
print("TOTAL: MAPE: %0.4f" % MAPE);

model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(np.array(X_array), Y)
Y_ransac = model_ransac.predict(X_array)
MSE = metrics.mean_squared_error(Y, Y_ransac)
RMSE = math.sqrt(MSE)
MAPE = mean_absolute_percentage_error(Y, Y_ransac);
print("RANSAC: TOTAL: MSE: %0.4f" % MSE);
print("RANSAC: TOTAL: RMSE: %0.4f" % RMSE);
print("RANSAC: TOTAL: MAPE: %0.4f" % MAPE);