#!/usr/bin/env python3

import math
import numpy as np
from scipy import stats
from scipy.stats import t

bilateral_5 = 1.96
bilateral_1 = 2.576

data_all = []
data = []
print("Carregando base...")
with open("Runner.txt") as f:
	for line in f:
		l = line.split("\t")
		l = list(map(lambda x:x.strip(), l))
		data_all.append(l);
		data.append([l[0], l[5]]);

data = np.array(data).astype(np.float64);
print(data);

print(stats.linregress(data));
slope, intercept, r_value, p_value, std_err = stats.linregress(data)
print("Equação linear: y = %0.4fx + %0.4f" % (slope, intercept))

print("Coeficiente Correlacao (Pearson): %0.4f" % r_value)
print("p_Value: %0.4f" % p_value)

pred2016 = slope*2016 + intercept;
print("Previsão 2016: %0.2f" % pred2016)

def rejectKendallNullHypothesis(data, alpha):
	tau, p_value = stats.kendalltau(data[:,0], data[:,1])
	N = len(data[:,0]);
	test = alpha*math.sqrt((2*(2*N+5))/(9*N*(N-1)))
	print("%0.4f > %0.4f" % (math.fabs(tau), test))
	return math.fabs(tau) > test
	
rejected = rejectKendallNullHypothesis(data, bilateral_5);
print("Kendall: Hipotese nula rejeitada(95%% de significancia): %s" % rejected);

rejected = rejectKendallNullHypothesis(data, bilateral_1);
print("Kendall: Hipotese nula rejeitada(99%% de significancia): %s" % rejected);

test_x = [-0.4326,-1.6656,0.1253,0.2877,-1.1465,1.1909,1.1892,-0.0376,0.3273,0.1746,-0.1867,0.7258,-0.5883,2.1832,-0.1364];
test_y = [-0.1898,0.3105,0.0783,-1.2667,-0.7478,1.1433,1.6583,-0.018,3.6235,-0.0826,-1.5764,2.6174,-4.2476,2.1379,-0.4186];
print(stats.linregress(test_x, test_y));

def rejectPearsonNullHypothesis(data, alpha):
	N = len(data[:,0]);
	slope, intercept, r_value, p_value, std_err = stats.linregress(data)
	tval = t.isf(alpha/2, df)
	test = (r_value*math.sqrt(N-2))/(math.sqrt(1-r_value**2))
	print("%0.4f > %0.4f" % (math.fabs(test), tval))
	return math.fabs(test) > tval
	
rejected = rejectKendallNullHypothesis(data, 0.05);
print("Pearson: Hipotese nula rejeitada(95%% de significancia): %s" % rejected);

rejected = rejectKendallNullHypothesis(data, 0.01);
print("Pearson: Hipotese nula rejeitada(95%% de significancia): %s" % rejected);
