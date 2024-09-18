
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
#from matplotlib.colors import Colormap
import math
#import cv2

#from scipy.signal import find_peaks
from copy import deepcopy

import lists

"""
def check_for_nonzero( img, center, mask ):

		centre [x,y]
		mask is n*m array, where n and m are both odd

	
	x = centre[0]
	y = centre[1]
	h_width = (mask[0]-1)/2
	h_hight = len(mask[0])
	
	for i in range( x-(m_width-1)/2, x+(m_width-1)/2 ):
		for j in range( x-m_hight, x+m_half_hight ):
			if img[i,j] :
				
	for i in range( x_width ):
		for j in range( x_hight ):
			x_ =
			if img[i,j] :
	
	return found
"""

def ToF2E(v,m): return((m*v**2)/2)


#%%-----------------------------------------------------------------------------
# read in data using Petters functions
#-----------------------------------------------------------------------------
fileName = '/home/rob/Desktop/Link to work-UU/DATA_ANALYSIS/analysis_ToF-ERDA/20230511_Studsvik/original_data/I-127_44MeV_ADOPT-1.lst'
#fileName = 'holder3/I_36MeV_SH3-07_screen.lst'
fileHeader,eventLists,timingList = lists.lstRead(fileName)
coin = [True, True, False, False, False, False, False, False]
zdrop = True
outLists = lists.getCoins(eventLists,coin,zdrop)
chn = [2048,2048]




#%%-----------------------------------------------------------------------------
# convert to 2d hist
#-----------------------------------------------------------------------------

binwidth = 2   ##### SET BINNING HERE #####
bins_x = int(chn[0]/binwidth)
bins_y = int(chn[1]/binwidth)

inds_multi = [ind for ind in range(8) if coin[ind]]
chmin = [0,0]
chmax = [chn[inds_multi[0]],chn[inds_multi[1]]]
nbins = [chn[inds_multi[0]]+1,chn[inds_multi[1]]+1]
nbins = [math.floor(nb/binwidth) for nb in nbins]
bins=[np.linspace(chmin[0],chmax[0],nbins[0]),np.linspace(chmin[1],chmax[1],nbins[1])]
print(f'channels used are {inds_multi}')
print(f'number of bins = {nbins}')
lhist,xed,yed = np.histogram2d(outLists[0],outLists[1],bins=bins)
#-----------------------------------------------------------------------------

#%%-----------------------------------------------------------------------------
# auto define bounds on the 2d hist, to reduce later processing
#-----------------------------------------------------------------------------


cut = 4


# create projections of the 2d histogram
pro_x = np.zeros(bins_x)
pro_y = np.zeros(bins_y)
for i in range(bins_x-1): pro_x[i] = sum(lhist[i,:])
for i in range(bins_y-1): pro_y[i] = sum(lhist[:,i])
# Scan backward from end of the projections, 
# if 5 consecutive bins sum to greater than 3,
# place the bound at the centre of the sum.
for i in reversed(range(bins_x-3)):
	if i == 3:
		bound_x = bins_x
		print('no good data found!')
		break
	if sum(pro_x[i-2:i+2]) > cut:
		print(i)
		bound_x = int(i*1.1)
		if bound_x >= bins_x:
			bound_x = bins_x
		break
for i in reversed(range(bins_y-3)):
	if i == 3:
		bound_y = bins_y
		print('no good data found!')
		break
	if sum(pro_y[i-2:i+2]) > cut:
		bound_y = int(i*1.1)
		if bound_y >= bins_y:
			bound_y = bins_y
		break
# create reduced 2d histogram
print(f'auto-bound on X = {bound_x}')
print(f'auto-bound on Y = {bound_y}')
bhist = lhist[0:bound_x,0:bound_y]
#-----------------------------------------------------------------------------





#%%-----------------------------------------------------------------------------
# apply lower threshold to remove noise events 
#-----------------------------------------------------------------------------
# create reduced 2d histogram

thresh_x = 2
thresh_y = 2

bthist = deepcopy(bhist)
bthist[0:bound_x,0:thresh_y] = 0
bthist[0:thresh_x,0:bound_y] = 0

# create projections of the 2d histogram
prot_x = np.zeros(bound_x)
prot_y = np.zeros(bound_y)
for i in range(bound_x-1): prot_x[i] = sum(bthist[i,:])
for i in range(bound_y-1): prot_y[i] = sum(bthist[:,i])
#-----------------------------------------------------------------------------





#-----------------------------------------------------------------------------
# plot everything so far
#-----------------------------------------------------------------------------
# Make colormap
customMap = mc.LinearSegmentedColormap.from_list(name='WKBGRY',
												 colors=[(1,1,1),(0,0,0),[1,1,0],[1,0,0]],
												 N = 200)

# Define the colors and their positions
colors = [
    (0.0, 'purple'),
    (0.2, 'blue'),
    (0.4, 'green'),
    (0.6, 'yellow'),
    (0.8, 'orange'),
    (1.0, 'red')
]

colors = [
    (0.0, 'blue'),
    (0.1, 'green'),
    (0.3, 'yellow'),
    (0.6, 'orange'),
    (1.0, 'red')
]

# Create the colormap
cmap = mc.LinearSegmentedColormap.from_list('my_map', colors)
cmap.set_under(color='white')  
"""
#%%
lextent = [yed[0],yed[-1],xed[0],xed[-1]]
# Plot original 2d hist
plt.figure()
plt.imshow(lhist.T,cmap=cmap,origin='lower',interpolation='none',vmin=0.1,extent=lextent)
plt.colorbar()
plt.plot([bound_x,bound_x],[0,bound_y],'r--')
plt.plot([0,bound_x],[bound_y,bound_y], 'r--')
plt.plot([thresh_x,thresh_x],[0,bound_y],'b--')
plt.plot([0,bound_x],[thresh_y,thresh_y], 'b--')
plt.grid(linestyle='--')
plt.xlabel('energy (channel)')
plt.ylabel('ToF (channel)')
plt.title('original')
# plot x-prjection
plt.figure()
plt.plot(pro_x)
plt.plot([bound_x,bound_x],[0,max(pro_x)*1.1],'r--')
plt.plot([thresh_x,thresh_x],[0,max(pro_x)*1.1],'b--')
plt.grid(linestyle='--')
plt.xlabel('energy (channel)')
plt.title('original')
# plot y-projection
plt.figure()
plt.plot(pro_y)
plt.plot([bound_y,bound_y],[0,max(pro_y)*1.1],'r--')
plt.plot([thresh_y,thresh_y],[0,max(pro_y)*1.1],'b--')
plt.grid(linestyle='--')
plt.ylabel('ToF (channel)')
plt.title('original')
"""
#%%

# plot bounded 2d hist
bextent = [0,bound_x,0,bound_y]
# Plot
plt.figure()
plt.imshow(bthist.T,cmap=cmap,origin='lower',interpolation='none',vmin=0.1,extent=bextent)
plt.colorbar(location='top')
plt.grid(linestyle='--')

#xticks = plt.xticks()
#plt.xticks(xticks * 2)


plt.xlabel('energy (channel)')
plt.ylabel('ToF (channel)')
#plt.title('bounded')
plt.tight_layout()




# plot x-prjection
plt.figure()
plt.plot(prot_x)
#plt.plot([bound_x,bound_x],[0,max(prot_x)*1.1],'r--')
#plt.plot([thresh_x,thresh_x],[0,max(prot_x)*1.1],'b--')
plt.xlim([0,bound_x])
plt.grid(linestyle='--')
plt.ylabel('counts')
# plot y-projection
plt.figure()
plt.plot(prot_y)
#plt.plot([bound_y,bound_y],[0,max(prot_y)*1.1],'r--')
#plt.plot([thresh_y,thresh_y],[0,max(prot_y)*1.1],'b--')
plt.xlim([0,bound_y])
plt.grid(linestyle='--')
plt.ylabel('counts')



#%%

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)



main_ax.imshow(bthist.T,cmap=cmap,origin='lower',interpolation='none',vmin=0.1,extent=bextent)
#main_ax.colorbar(location='top')
main_ax.grid(linestyle='--')
#main_ax.xlabel('energy (channel)')
#main_ax.ylabel('ToF (channel)')
#main_ax.tight_layout()

# plot x-prjection
x_bins = np.linspace(0,bound_x,bound_x)
x_hist.step(x_bins, prot_x)
x_hist.invert_yaxis()
#x_hist.xlim([0,bound_x])
#x_hist.grid(linestyle='--')
#x_hist.ylabel('counts')

# plot y-projection
y_bins = np.linspace(0,bound_y,bound_y)
y_hist.step(y_bins, prot_y)
y_hist.invert_yaxis()
#y_hist.xlim([0,bound_y])
#y_hist.grid(linestyle='--')
#y_hist.ylabel('counts')




"""
#%%



# plot bounded 2d hist
# Plot
plt.figure()
plt.imshow(bthist.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.colorbar()
plt.grid(linestyle='--')
plt.xlabel('energy (channel)')
plt.ylabel('ToF (channel)')
plt.title('bounded with lower threshold')
#-----------------------------------------------------------------------------




#%%

# log the image
loged = np.log(bhist+1)
plt.figure()
plt.imshow(loged.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.colorbar()
plt.title('loged')

loged = bhist # unlog the image (for testing)


kernel = np.ones((5,5),np.float32)/25


kernel = np.array([[0,0,0,0,0,0,0,1,1],
				   [0,0,0,0,0,1,1,1,1],
				   [0,0,0,1,1,1,1,1,0],
				   [0,0,1,1,1,1,1,1,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,1,1,1,1,1,1,0,0],
				   [0,1,1,1,1,1,0,0,0],
				   [1,1,1,1,0,0,0,0,0],
				   [1,1,0,0,0,0,0,0,0]])/81

kernel = np.array([[1,1,0,0,0,0,0,0,0],
				   [1,1,1,1,0,0,0,0,0],
				   [0,1,1,1,1,1,0,0,0],
				   [0,1,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,1,0],
				   [0,0,0,1,1,1,1,1,0],
				   [0,0,0,0,0,1,1,1,1],
				   [0,0,0,0,0,0,0,1,1]])/81


kernel = np.array([[0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0],
				   [0,0,1,1,1,1,1,0,0]])/81

kernel = np.array([[0,0,0,0,0,0,0,0,0],
				   [0,0,0,0,0,0,0,0,0],
				   [1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1],
				   [1,1,1,1,1,1,1,1,1],
				   [0,0,0,0,0,0,0,0,0],
				   [0,0,0,0,0,0,0,0,0]])/81

#blur = cv2.filter2D(loged,-1,kernel)


#blur the image
blur = cv2.GaussianBlur(loged,(25,25),0)


plt.figure()
plt.imshow(blur.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.colorbar()
plt.title('filtered')


#%%

# Normalize the image
img_norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img_norm = img_norm.astype(np.uint8)
plt.figure()
plt.imshow(img_norm.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.colorbar()
plt.title('normalised')


plt.figure()
plt.hist(img_norm.ravel(), bins=256) #calculating histogram
plt.yscale('log')
plt.title('pixel intensity')


# apply binary thresholding
ret, thresh = cv2.threshold(img_norm, 25, 255, cv2.THRESH_BINARY)
thresh=thresh.astype(np.uint8)
plt.figure()
plt.imshow(thresh.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.colorbar()
plt.title('binary')

#%%
# find all of the connected components (white blobs in your image).
# im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(thresh)
# stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
# here, we're interested only in the size of the blobs, contained in the last column of stats.
sizes = stats[:, -1]
# the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
# you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
sizes = sizes[1:]
nb_blobs -= 1

# minimum size of particles we want to keep (number of pixels).
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
min_size = 81  

# output image with only the kept components
im_strip = np.zeros_like(im_with_separated_blobs)
# for every component in the image, keep it only if it's above min_size
for blob in range(nb_blobs):
    if sizes[blob] <= min_size:
        # see description of im_with_separated_blobs above
        im_strip[im_with_separated_blobs == blob + 1] = 255

plt.figure()
plt.imshow(im_strip.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.title('stripped')


#%%

plt.figure()
plt.plot(img_norm[280])

plt.figure()
plt.plot(bhist[280])

peaks=find_peaks(img_norm[280])
print(peaks)

#%%
peaks=[]
for x in blur:
	p,d = find_peaks(x)
	peaks.append(p)

p_array_x = np.zeros([bound_x,bound_y])

for x in range(len(peaks)):
	for y in peaks[x]:
		p_array_x[x,y] = 1

plt.figure()
plt.imshow(p_array_x.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.title('peak positions')

peaks=[]
for i in range(len(blur[0])):
	p,d = find_peaks(blur[:,i])
	peaks.append(p)

p_array_y = np.zeros([bound_x,bound_y])

for y in range(len(peaks)):
	for x in peaks[y]:
		p_array_y[x,y] = 1

plt.figure()
plt.imshow(p_array_y.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.title('peak positions')

new_array = (p_array_x + p_array_y) -1
plt.figure()
plt.imshow(new_array.T,cmap=customMap,origin='lower',interpolation='none',vmin=0,extent=bextent)
plt.title('peak positions')

"""



