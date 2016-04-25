#!/usr/bin/env python
# Input: a text file with a ticker on each line.
# Dependends on NUMPY and PANDAS

import csv, math, os, shutil, datetime, urllib, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from sys import argv

# Tickers is a text file that contains all the tickers you want to analyze. 
# Start is the year you'd like to start on
script, txt = argv
now = datetime.datetime.now()
#matplotlib.style.use('ggplot')

# Parameters:

#Start year
start = 2014

#start = datetime.datetime(start, 1, 1)

print "#######################################################"
print "Beginning simulation starting in " + str(start)

if os.path.isdir('csvfiles'):
	shutil.rmtree('csvfiles')
os.mkdir('csvfiles')

# Step 1: record tickers
# This simple chunk opens the "txt" file containing tickers and records all of the tickers in a list
def get_tickers(textfile):
	tickers = []
	with open(textfile) as t:
		tickers = t.readlines()
	tickers = [x.strip('\n') for x in tickers]
	for t in tickers:
		url = 'http://real-chart.finance.yahoo.com/table.csv?s=' + t + '&a=01&b=1&c=' + str(start) + '&d=' + str(now.month-1) + '&e=' + str(now.day) + '&f=' + str(now.year) + '&g=d&ignore=.csv'
		csvfile = urllib.URLopener()
		csvfile.retrieve(url, './csvfiles/' + t + '.csv')
	return tickers

########################################################################################################
# Establish indicators and other important numbers:

# Moving average. Self explanatory. Data is a pandas dataframe, period is an integer
	# Apparently 9 and 13 are special numbers for this
def moving_average(data, period, type='simple'):
	"""
	Compute an n period moving average.
	type is 'simple' or 'exponential'
	"""
	try:
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)

	if type == 'simple':
		weights = np.ones(period)
	else:
		weights = np.exp(np.linspace(-1., 0., period))

	weights /= weights.sum()

	a = np.convolve(x, weights, mode='full')[:len(x)]
	a[:period] = a[period]
	return a

# Bollinger curves. Data is a pandas dataframe, period is an integer, multiplier is how many standard deviations to use. Usually people use 2 and -2
def bollinger(data, period, multiplier):
	try:
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)
	return pd.rolling_mean(x, period) + multiplier*(pd.rolling_std(data['Adj Close'], period, min_periods=period))

# Relative Strength Index. Data is a pandas dataframe, period is an integer.
def rsi(data, period):
	try:
		delta = data['Adj Close'].diff()
	except:
		delta = data.diff()

	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = pd.rolling_mean(dUp, period)
	RolDown = pd.rolling_mean(dDown, period).abs()

	RS = RolUp / RolDown
	rsi = 100 - (100.0 / (1.0 + RS))
	return rsi

# MACD is apparently good. Here it is. No idea how to use it :D
def moving_average_convergence(data, nslow=26, nfast=12):
	"""
	Compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(data) arrays
	"""
	try:
		x = np.asarray(data['Adj Close'])
	except:
		x = np.asarray(data)
	emaslow = moving_average(x, nslow, type='exponential')
	emafast = moving_average(x, nfast, type='exponential')
	return emafast - emaslow

################################################################################################################
# Normalize data:
# Normalizes any data in a numpy array and returns it as a numpy array :D
# Formula: element - mean / (max - min)
def normalize(nparray):
	return ((nparray - nparray.min()) / (nparray.max() - nparray.min()))

# Step 2: Analyze/organize data. For any indicators you don't plan on using, feel free to comment that shiet out
def analyze_organize(ticker):
	tickerdata = pd.read_csv('./csvfiles/' + ticker + '.csv')
	tickerdata = tickerdata.iloc[::-1]
	# Establishing indicators
	tickerdata['MACD'] = moving_average_convergence(tickerdata)
	tickerdata['MACD_Signal'] = moving_average(tickerdata['MACD'], 9, 'exponential')
	tickerdata['MACD_Indicator'] = tickerdata['MACD'] - tickerdata['MACD_Signal']
	tickerdata['RSI14'] = rsi(tickerdata, 14)
	del tickerdata['Open']
	del tickerdata['High']
	del tickerdata['Low']
	del tickerdata['MACD']
	del tickerdata['MACD_Indicator']
	return tickerdata

# This just runs all the functions. BANANAS HAVE POTASSIUM
def main(tickers):
	tickerlist = []
	rsilist = []
	macdlist = []
	for t in tickers:
		print "Processing data for " + t
		data = analyze_organize(t)
		print data.tail(n=5)
		print '_________________________________________________________________\n'
	# 	tickerlist.append(t)
	# 	rsilist.append(data['RSI14'][-1])
	# 	macdlist.append(data['MACD_Indicator'][-1])
	# print STOCK, RSI, MACD
	# for x in range(0, len(tickerlist)):
	# 	print tickerlist[x], rsilist[x], macdlist[x]


main(get_tickers(txt))
#shutil.rmtree('csvfiles')


# """ TO DO """
# """ - Filling the position array between buy/sell
# 	- Multiple indicators (using AND)
# 	- Weighting different indicators (normalizing to 1?)
# 	- Scikit learn?
# 	"""