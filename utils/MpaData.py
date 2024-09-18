# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:47:24 2023

@author: thets185
"""

from enum import Enum
import numpy as np
#==============================================================================
# Introducing an enumerator for more efficient condition checks
class ReadStatus(Enum):
	HEADER = 0
	DATA = 1
	CDAT = 2
#==============================================================================


class DATA:
	def __init__(self):
		self.id = ''
		self.xbins = 0
		self.ybins = 0
		self.rt=0
		self.lt=0
		self.cnts=0
	def set_id(self,id_num):
		self.id = id_num
	def set_xbins(self,xbins):
		self.xbins = xbins
	def set_ybins(self,ybins):
		self.ybins = ybins
	def set_rt(self,realtime):
		self.rt = realtime
	def set_lt(self,livetime):
		self.lt = livetime
	def set_cnts(self,cnts):
		self.cnts=cnts

def read_MPA(mpaFilePath):
	"""
	Constructor of MpaData object

	Parameters
	----------
	mpaFilePath : String/Path
		Path of the .mpa file to be read.

	Returns
	-------
	None.

	"""

	DATA_list = []
	CDAT_list = []
	list_pos = 0
	clist_pos = 0

	dataFile = open(mpaFilePath, 'r')
		
	for line in dataFile:										# Loop through each line of the input data file
	

	### read from HEADER ##################################################
		
		if( line[0:4]=="[ADC"): # record the number of channels for each ADC
			adc_id = line[4]
			adc_len = int(dataFile.readline().strip().split('=')[1])
			if int(dataFile.readline().strip()[-1])!=0:										# check the adc is active
				ADC = DATA()																# create data object for ADC
				ADC.set_id('ADC'+adc_id)													# set the adc number
				ADC.set_xbins(adc_len)														# set the range
				while line[0:9]!='realtime=': line=dataFile.readline()						# find the realtime line
				ADC.set_rt(float(line[9:-1]))												# append the realtime
				while line[0:9]!='livetime=': line=dataFile.readline()						# find the livetime line
				ADC.set_lt(float(line[9:-1]))												# append the livetime
				DATA_list.append(ADC)														# append the object to te list
		
		if( line[0:4]=="[MAP"): # record the number of channels for each ADC
			map_id = line[4]
			MAP = DATA()																	# create data object for ADC
			MAP.set_id('MAP'+map_id)														# set the map number
			while line[0:6]!='range=': line=dataFile.readline()								# find the realtime line
			maplen=int(line[6:-1])
			while line[0:5]!='xdim=': line=dataFile.readline()								# find the realtime line
			xdim=int(line[5:-1])
			MAP.set_xbins(xdim)
			MAP.set_ybins(int(maplen/xdim))
			CDAT_list.append(MAP)															# append the object to te list

	### read DATA #########################################################
		
		if( line[0:5]=="[DATA"):
			data = []
			chanCounter=0
			while chanCounter < DATA_list[list_pos].xbins:
				data.append(int(dataFile.readline().strip()))
				chanCounter+=1
			DATA_list[list_pos].set_cnts(np.array(data))
			list_pos+=1

		if( line[0:5]=="[CDAT"):
			xdata = []
			data = []
			chanCounter=0
			xchanCounter=0
			while chanCounter < CDAT_list[clist_pos].ybins:
				while xchanCounter < CDAT_list[clist_pos].xbins:
					xdata.append(int(dataFile.readline().strip()))
					xchanCounter+=1
				data.append(xdata)
				chanCounter+=1
			CDAT_list[clist_pos].set_cnts(np.array(data))
			clist_pos+=1

	return DATA_list, CDAT_list
#------------------------------------------------------








#print(mpaFile.ADC2)
#mpaFile.plotSpectra()
# print(mpaFile.getROI(0,646,718))
# print(mpaFile.getROI(0,298,443))
# print(mpaFile.getROI(0,10,30))
