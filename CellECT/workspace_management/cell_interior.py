# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import scipy as sp
from numpy import histogram


def histeq(im,nbr_bins=256):
	"""
	Histogram equalization function from Solem's Vision Blog at
	http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
	"""
	#get image histogram
	imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
	cdf = imhist.cumsum() #cumulative distribution function
	cdf = 255 * cdf / cdf[-1] #normalize

	#use linear interpolation of cdf to find new pixel values
	im2 = np.interp(im.flatten(),bins[:-1],cdf)

	return im2.reshape(im.shape), cdf



class CellInterior(object):

	def __init__(self, volume):

		self.vol = volume
		
	def run_cell_centroid_estimator(self):

		hist = histogram(self.vol)
		
		print "HERE"

		
