#!/usr/bin/env python
# Input: a text file with a ticker on each line.
# Dependends on NUMPY and PANDAS

import csv, math, os, shutil, datetime, urllib, matplotlib
import pandas as pd
from pandas_datareader import data, wb
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
	tickerdata['SMA200'] = moving_average(tickerdata, 200, 'simple')
	tickerdata['SMA50'] = moving_average(tickerdata, 50, 'simple')
	tickerdata['Bol_High_20']=bollinger(tickerdata, 20, 2)
	tickerdata['Bol_Low_20']=bollinger(tickerdata, 20, -2)
	tickerdata['RSI14'] = rsi(tickerdata, 14)

	# Normalizing everything
	# for x in tickerdata:
	# 	if x != 'Date':
	# 		tickerdata[x] = normalize(tickerdata[x])
	return tickerdata

# This is where the actual algorithm goes. Use whatever indicators you want! :D
	# Position is either 1 (long), or 0 (short).
def backtest(data, budget):
	# This is where shit gets organised and the "position" array is developed.
	position = np.digitize( (data['RSI14']), [0,30,70,100])
	position = np.diff(position)
	print position
	shit = np.zeros( data['Adj Close'].shape)
	shit = data['Adj Close'][position == 1]
	shit = data['Adj Close'][position ==-1]

	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax1.plot(data['Adj Close'])
	ax2 = fig.add_subplot(212)
	ax2.plot(data['RSI14'])
	plt.show()
	#print position
	# Here's the backtesting part. I know it's a cop out using ZIP and turning it into a list thing
	# I'll try to figure out an alternative way of handling the data 
	money = float(budget)
	stocks = 0
	for price, pos in zip( data['Adj Close'], position):
		if pos == 1 and stocks == 0:
			stocks = int(money/price)
			money -= stocks*price
			#print "Bought " + str(stocks)
		elif pos == -1 and stocks != 0:
			#print "Sold " + str(stocks)
			money += stocks*price
			stocks = 0
	return money + data['Adj Close'].iloc[-1] * stocks

# This just runs all the functions. BANANAS HAVE POTASSIUM
def main(tickers):
	for t in tickers:
		print "____________________________________________________________________\n"
		print "Processing data for " + t
		data = analyze_organize(t)
		budget = 10000
		print "Budget = $" + str(budget)
		algo = backtest(data, budget)
		gains = round(algo-budget, 2)
		long_gains = round((data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) * int(budget/(data['Adj Close'].iloc[0])), 2)

		print "Algorithm yield:    " + str(gains) + ", representing change of " + str( round(100*(gains) / budget, 2) ) + "%"
		print "Holding long yield: " + str(long_gains) + ", representing change of " + str( round(100*(long_gains) / budget, 2) ) + "%"
		print "____________________________________________________________________\n"


main(get_tickers(txt))
#shutil.rmtree('csvfiles')


# """ TO DO """
# """ - Filling the position array between buy/sell
# 	- Multiple indicators (using AND)
# 	- Weighting different indicators (normalizing to 1?)
# 	- Scikit learn?
# 	"""