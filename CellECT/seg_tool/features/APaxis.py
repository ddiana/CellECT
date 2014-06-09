import numpy as np
import scipy as sp
import pylab
import pdb
from scipy.interpolate import UnivariateSpline



class APaxis(object:


	def __init__(self,x,y,z, xres, yres, zres, vol_size, mask = None):

		self.xres = xres
		self.yres = yres
		self.zres = zres
		self.vol_size = vol_size

		self.make_axis(x,y,z)
		self.begin = None
		self.end = None

		if not vol is None:
			self.mark_begin_and_end(mask)



	def make_axis(self,x,y,z):

		x = np.array(x) * self.xres
		y = np.array(y) * self.yres
		z = np.array(z) * self.zres

		self.function_xy = interpolate.UnivariateSpline(x, y, k =3)
		self.function_xz = interpolate.UnivariateSpline(x, z, k =3)

		self.x = np.arange(self.vol_size[0])
		self.y = self.function_xy(self.x)
		self.z = self.function_xz(self.z)

	def mark_begin_and_end(self, mask):

		ind = 0
		while ind < len(self.x) and mask[self.x[ind], self.y[ind], self.z[ind]] == 0:
			ind+=1

		if ind < len(self.x):
			self.begin = ind

		ind = len(self.x)

		while ind >=0 and mask[self.x[ind], self.y[ind], self.z[ind]] == 0:
			ind-=1

		if ind >= 0:
			self.end = ind

		
	def add_segment_projection_properties(self, segment):

		centroid = segment.feature_dict["centroid_res"]

		dist_to_axis = ((self.x - centroid[0])**2 + (self.y - centroid[1])**2 + (self.z - centroid[2])**2 ) ** 0.5

		closest_pos = np.argmin(dist_to_axis)
		
		segment.add_feature("dist_to_AP_axis", dist_to_axis[closest_pos])

		position_along_axis = float(closest_pos - self.begin) / (self.end - self.begin) * 100
		segment.add_feature("")




