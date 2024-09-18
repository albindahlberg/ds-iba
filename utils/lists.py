#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:24:39 2023

@author: frost
"""
import struct
import numpy as np

#---------------------------#
#	  Read .lst file	   #
#---------------------------#

def lstRead(fileName):
	# Opens binary lst file and returns its content as a list of
	# eight lists. Each sublist corresponds to an ADC, and each
	# element to an event. A None entry means that the ADC did not
	# trigger for the event. The timingList output gives the number
	# of events for each millisecond, and since eventLists is
	# chronological, it can be used to only retreive events between
	# certain time indices.
	with open(fileName, mode='rb') as f:
		fileContent = f.read()
	
	try:
		# Everything before [LISTDATA] is read as header
		fileHeader, fileContent = fileContent.split(b'[LISTDATA]')
		fileHeader = fileHeader.decode('windows-1252')
		# Remove first two characters from fileContent (carriage return and newline)
		fileContent = fileContent[2:]
	except ValueError:
		print('Warning! File header not found: reading entire file as data.')
		fileHeader = 'File header not found!'
	
	# Split content at:
	# \x00@\xff\xff\xff\xff (16384,65535,65535 =
	# timer event high word followed by data). Yields
	# millisecond event groups. Each such group also
	# contains all following timer events without data.
	fileContent = fileContent.split(b'\x00@\xff\xff\xff\xff')
	
	# If the first four bytes is an synchronize mark,
	# modify fileContent[0] to read the following data.
	# NOTE: if fileContent[0] is only b'\xff\xff'
	# (timer event low word), that is to say if the
	# list data starts with a timer event, then the
	# if statement criterion is evaluated as False
	# => no index out of bounds error. This is the
	# desired functionality.
	if fileContent[0][0:4] == b'\xff\xff\xff\xff':
		fileContent[0] = fileContent[0][4:]
	
	# Add timer event low word, all ADC:s active indicator
	# to mark the end of the last event
	fileContent[-1] += b'\xff\xff'
	
	# Declaration of output lists
	eventLists = [[], [], [], [], [], [], [], []]
	timingList = []
	
	# Declarations of variables
	nsub = len(fileContent)
	nints = len(max(fileContent, key=len))
	j = 0
	k = 0
	n = 0
	nloop = 0
	nempty = 0

	# Allocate enough memory for longest event group
	cyt_filec = np.zeros(nints, dtype=int) #<unsigned int *>malloc(nints*cython.sizeof(int))

	i = 0
	strunp = struct.unpack

	try:
		while i < nsub:
			
			# Read one event group into cyt_filec with struct.unpack
			tempdat = strunp('H'*(len(fileContent[i])//2),fileContent[i])

			# Copy event group to cython array. Finally, nf is the number
			# of elements in the event group (including flag bytes).
			nf = 0
			for item in tempdat:
				cyt_filec[nf] = item
				nf+=1

			# Work out the event group
			j = 0
			nloop = 0

			# Check if current element in event group is a timer event low word,
			# (larger than 0b1111111100000000, which is the lowest value if all
			# ADCs are dead). If it is, stop reading events.
			# Something like [while j < nf-1] does not work, since timer events
			# without data in that case will not terminate the reading of event
			# data. OBS! It is possible here to extract the dead ADC flags for
			# the next millisecond event group from cyt_filec[j]. This information
			# is presently discarded.
			while cyt_filec[j] < 65280:
				n = 0

				# Check for dummy word
				if cyt_filec[j+1] & 32768 != 0:
					n+=1
				# Append eventLists based on active ADCs
				
				for k in range(8):
					if cyt_filec[j] & 2**k != 0:
						eventLists[k].append(cyt_filec[j+2+n])
						n+=1
					else:
						eventLists[k].append(None)
				j+=n+2
				nloop+=1
				
			timingList.append(nloop)

			# Check if the millisecond event group is followed by any
			# timer events without data. If that is the case, add
			# milliseconds with zero events to timingList.
			nempty = (len(tempdat)-j-1)/2
			j = 0
			while j < nempty:
				timingList.append(0)
				j+=1

			i+=1
	
	except:
		print('PROBLEM!!!')
	
	finally:
		del fileContent
		return(fileHeader,eventLists,timingList)


###############################################################################

def getCoins(eventLists,coin,zdrop):
	# Returns all events in eventLists for which all the ADCs
	# specified in coin are active. For example, if coin =
	# = [T T F F F F F F], ADCs 1 and 2 are required to be
	# in coincidence. An event in channel zero of a given ADC is
	# considered as an active event for the ADC in question if
	#"drop zero" is unchecked. If "drop zero" is checked, a zero
	# event is not considered as "coincident".
	inds = [ind for ind, ADC in enumerate(coin) if ADC]
	outLists = [[],[],[],[],[],[],[],[]]
	if zdrop:
		for j in range(len(eventLists[inds[0]])):
			add = True
			for ind in inds:
				if eventLists[ind][j] is None or eventLists[ind][j] == 0:
					add = False
			if add:
				for i in range(8):
					outLists[i].append(eventLists[i][j])
	else:
		for j in range(len(eventLists[inds[0]])):
			add = True
			for ind in inds:
				if eventLists[ind][j] is None:
					add = False
			if add:
				for i in range(8):
					outLists[i].append(eventLists[i][j])

	return(outLists)



###############################################################################

def getACoins(eventLists,coin,zdrop):
	# ANTICOINCIDENCE
	# Returns all events in eventLists where the ADCs
	# specified in coin are NOT active. For example
	# if coin = [T T F F F F F F], ADCs 1 and 2 are required
	# NOT to have triggered. If zero drop is checked, an event in
	# channel zero is considered "not coincident". If zero drop is
	# not checked, an even in channel zero is considered "coincident".
	# The zero drop checkbox thus has the same meaning for this
	# anticoincidence function as it does for the coincidence function.
	inds = [ind for ind, ADC in enumerate(coin) if ADC]
	outLists = [[],[],[],[],[],[],[],[]]
	if zdrop:
		for j in range(len(eventLists[inds[0]])):
			add = True
			for ind in inds:
				if not (eventLists[ind][j] is None or eventLists[ind][j] == 0):
					add = False
			if add:
				for i in range(8):
					outLists[i].append(eventLists[i][j])
	else:
		for j in range(len(eventLists[inds[0]])):
			add = True
			for ind in inds:
				if not eventLists[ind][j] is None:
					add = False
			if add:
				for i in range(8):
					outLists[i].append(eventLists[i][j])

	return(outLists)


###############################################################################
###############################################################################
###############################################################################
