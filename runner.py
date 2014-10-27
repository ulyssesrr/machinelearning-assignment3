#!/usr/bin/env python3

import numpy as np
from scipy import stats

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
print("Coeficiente Correlacao (Pearson): %0.4f" % r_value)
print("p_Value: %0.4f" % p_value)

pred2016 = slope*2016 + intercept;
print("Previsão 2016: %0.2f" % pred2016)


tau, p_value = stats.kendalltau(data[:,0], data[:,1])
print("Kendall: Tau: %0.4f P: %0.4f" % (tau, p_value))
