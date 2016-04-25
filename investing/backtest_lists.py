#!/usr/bin/env python
# This script takes a text file and spits out a bunch of graphs of the moving averages 
# Input: a text file with a ticker on each line.
# Dependencies:
	# Matplotlib for graphing and organizing data. Can be installed with: yum/apt-get install matplotlib or pip install matplotlib

import csv, math, os, shutil, datetime, urllib
from sys import argv

# Tickers is a text file that contains all the tickers you want to analyze. 
# Start is the year you'd like to start on
script, txt = argv
now = datetime.datetime.now()

# Parameters:

#Start year
start = 2010

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
	return tickers

########################################################################################################
# Establish indicators and other important numbers:

# Moving average. Self explanatory
	# Apparently 9 and 13 are special numbers for this (for days)
def movingaverage(prices, days):
	ma = []
	for x in range(0, len(prices)):
		if x <= days:
			ma.append( (prices[x]) )
		if x > days:
			ma.append( sum(prices[x-days:x-1])/days )
	return ma

# VWAP is the 'volume weighted average price', a great way to correlate volume and price. One of the best indicators imo
def vwap(prices, days, volume):
	vwap = []
	for x in range(0, len(prices)):
		if x <= days:
			vwap.append( (prices[x] / volume[x]) )
		if x > days:
			vwap.append( sum(prices[a] * volume[a] for a in range(x-days, x-1)) / sum(volume[x-days:x-1]) )
	return vwap

# This is the simple (possibly effective?) algorithm presented by Keith Fitz. Found here:
#       http://totalwealthresearch.com/find-profits-in-any-market-with-keith-fitz-geralds-x-pattern/?from=oo
def bs_low(prices, days):
	bs = []
	for x in range(0, len(prices)):
		if x <= days:
			bs.append( 0.5 )
		else:
			bs.append(0.05 + 0.9*((days - prices[x-days:x].index(max(prices[x-days:x])))/float(days)))
	return bs

def bs_high(prices, days):
	bs = []
	for x in range(0, len(prices)):
		if x <= days:
			bs.append( 0.5 )
		else:
			bs.append(0.05 + 0.9*((days - prices[x-days:x].index(min(prices[x-days:x])))/float(days)))
	return bs

# Find out when two equal-length lists. In this case, you want list 1 to be bigger than list 2 in order to buy
def find_convergence(list1, list2, threshold):
	mylist = []
	for x in range(0, len(list1)):
		if list1[x]*(1 - threshold) < list2[x]:
			mylist.append(0)
		elif list1[x]*(1 + threshold) > list2[x]:
			mylist.append(1)
	return mylist


##################################################################################################################################################################

# Step 2: retrieve data. Saves all yahoo finance ticker info into a folder
def main(tickers):
	for t in tickers:
		print "________________________________________\n"
		print "Processing data for " + t
		url = 'http://real-chart.finance.yahoo.com/table.csv?s=' + t + '&a=01&b=1&c=' + str(start) + '&d=' + str(now.month-1) + '&e=' + str(now.day) + '&f=' + str(now.year) + '&g=d&ignore=.csv'
		csvfile = urllib.URLopener()
		csvfile.retrieve(url, './csvfiles/' + t + '.csv')



	# Step 3: organize data
		# Day count. NOT the day/month/year version
		days = []
		# Date of price/volume/technicals. This is the day/month/year versoin
		date = []
		# Closing price after split/AH adjustments
		adj_close_price = []
		# Volume of trades for the day
		volume = []

		day = 1
		with open('./csvfiles/' + t + '.csv', 'rb') as csvfile:
			sreader = csv.reader(csvfile)
			for row in sreader:
				if str(row[0]) == 'Date':
					pass
				else:
					date.append(str(row[0]))
					volume.append(float(row[5]))
					adj_close_price.append(round(float(row[6]), 2))
					days.append(day)
					day += 1

		# Data is now recorded, reverse lists
		adj_close_price = list(reversed(adj_close_price))
		volume = list(reversed(volume))
		date = list(reversed(date))

		#These two call the algorithm
		iteration = []
		for x in range(1, 200):
			iteration.append(x)
		results = []
		for i in iteration:
			profit = algorithm(find_convergence( adj_close_price, movingaverage(adj_close_price, i), 0 ), adj_close_price)
			#print "Profit at period of " + str(i) + ": " + str(profit)
			results.append( profit )
		print "Profit from holding 20 positions long: " + str(20*(adj_close_price[-1] - adj_close_price[0]))
		print "Highest yield is " + str(max(results)) + " from period of " + str(iteration[results.index(max(results))])
		print "Algorithm yielded " + str(round(100*(max(results)/(20*adj_close_price[0])), 2)) + "% gain. Holding yielded " + str(round(100*((20*(adj_close_price[-1] - adj_close_price[0]))/(20*adj_close_price[0])), 2)) + "% gain."

		#########################################################################################
		# Here is where the actual algorithm is placed

def algorithm(indicator, prices):
	profit = 0
	buy = 0
	stocks = 0
	for i in range(0, len(indicator)):
		if indicator[i] == 1 and stocks == 0:
			buy += prices[i] * 20
			stocks = 20
		elif indicator[i] == 0 and stocks != 0:
			profit += prices[i]*stocks - buy
			buy = 0
			stocks = 0
		elif i == len(indicator) and stocks != 0:
			profit += prices[i]*stocks - buy
			print "Holding at end."

	return profit


main(get_tickers(txt))
shutil.rmtree('csvfiles')
