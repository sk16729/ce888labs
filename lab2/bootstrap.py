import matplotlib
matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np 
import math




def boostrap(statistic_func, iterations, data):
	samples  = np.random.choice(data,replace = True, size = [iterations, len(data)])
	#print samples.shape
	data_mean = data.mean()
	vals = []
	for sample in samples:
		sta = statistic_func(sample)
		#print sta
		vals.append(sta)
	b = np.array(vals)
	#print b
	lower, upper = np.percentile(b, [2.5, 97.5])
	return data_mean,lower, upper



if __name__ == "__main__":
	df = pd.read_csv('./vehicles.csv')
	#print df.columns
	
	data = df.values.T[1]
	
	boot = boostrap(np.mean, 100000, data)
	print(boot)
	
	v=0.0
	
	for i in data:
		v=v+((i-boot[0])**2)
	
	var=v/data.size
	s=math.sqrt(var)
	
	print("var:" +str(var))
	print("standard deviation:" +str(s))
	
		
	


	