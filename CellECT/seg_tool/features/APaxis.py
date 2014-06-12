import numpy as np
import scipy as sp
import pylab
import pdb
from scipy.interpolate import UnivariateSpline



class APaxis(object):


	def __init__(self,x,y,z, xres, yres, zres, vol_size, mask = None):

		self.xres = xres
		self.yres = yres
		self.zres = zres
		self.vol_size = vol_size

		self.make_axis(x,y,z)
		self.begin = None
		self.end = None

		if not mask is None:
			self.mark_begin_and_end(mask)

#		self.tempx = []
#		self.tempy = []
#		self.tempz = []


	def get_samples_on_line (self, x1,y1,z1, x2,y2,z2):


		length =  ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 ) ** 0.5
	
		sample_size = self.xres*5

		number_samples = int(length/ sample_size)
		
		points_x = []
		points_y = []
		points_z = []

		delta_x = (x2-x1) / float(number_samples)
		delta_y = (y2-y1) / float(number_samples)
		delta_z = (z2-z1) / float(number_samples)

		for i in xrange(number_samples):

			new_x = x1 + i*delta_x
			new_y = y1 + i*delta_y
			new_z = z1 + i*delta_z			

			points_x.append(new_x)
			points_y.append(new_y)
			points_z.append(new_z)		
	
		return points_x, points_y, points_z
		
		

	def make_axis(self,x,y,z):

		self.x = []
		self.y = []
		self.z = []


		x = np.array(x) * self.xres
		y = np.array(y) * self.yres
		z = np.array(z) * self.zres


		for i in xrange(1,len(x)):

			list_x, list_y, list_z = self.get_samples_on_line (x[i-1],y[i-1],z[i-1], x[i],y[i],z[i])
			self.x.extend(list_x)
			self.y.extend(list_y)
			self.z.extend(list_z)


		self.x = np.array(self.x)
		self.y = np.array(self.y)
		self.z = np.array(self.z)


	def mark_begin_and_end(self, mask):

		# find where along the axis the animal begins and ends
		self.begin = 0
		self.end = len(self.x)

#		pdb.set_trace()
#		ind = 0
#		while ind < len(self.x) and mask[self.x[ind], self.y[ind], self.z[ind]] == 0:
#			ind+=1
#		if ind < len(self.x):
#			self.begin = ind
#		ind = len(self.x)
#		while ind >=0 and mask[self.x[ind], self.y[ind], self.z[ind]] == 0:
#			ind-=1
#		if ind >= 0:
#			self.end = ind



	def get_line_tangent_at_pos(self, projection_pos):

		a = None
		b = None
		if projection_pos >0 and projection_pos < len(self.x)-1:
			a = projection_pos -1
			b = projection_pos +1
		elif projection_pos==0:
			a = projection_pos
			b = projection_pos +2
		elif projection_pos == len(self.x)-1:
			a = projection_pos -2
			b = projection_pos

		x1 = self.x[a]
		y1 = self.y[a]
		z1 = self.z[a]

		x2 = self.x[b]
		y2 = self.y[b]
		z2 = self.z[b]

		x = (x2-x1)
		y = (y2-y1)
		z = (z2-z1)

		norm = (x**2 + y**2 + z**2)**0.5
		x = x/norm
		y = y/norm
		z = z/norm

		return (x,y,z)


		
	def add_segment_projection_properties(self, segment):


		centroid = segment.feature_dict["centroid_res"]

		dist_to_axis = ((self.x - centroid[0])**2 + (self.y - centroid[1])**2 + (self.z - centroid[2])**2 ) ** 0.5

		closest_pos = np.argmin(dist_to_axis)
		
		segment.add_feature("dist_to_AP_axis", dist_to_axis[closest_pos])

		position_along_axis = float(closest_pos - self.begin) / (self.end - self.begin) * 100
		segment.add_feature("position_along_AP_axis", position_along_axis)

		segment_unit_vector = segment.feature_dict["line_fit"][:3]
		APaxis_unit_vector = self.get_line_tangent_at_pos(closest_pos)

		aa = segment_unit_vector
		bb = APaxis_unit_vector		

		norm = lambda x: (x[0]**2 + x[1]**2 + x[2]**2)**0.5

		angle = np.arctan2(norm(np.cross(aa,bb)),np.dot(aa,bb))* 180/ np.pi

		if angle > 90:
			angle = 90 - np.abs(90-angle)

#		self.tempx.append(centroid[0])
#		self.tempy.append(centroid[1])
#		self.tempz.append(centroid[2])

####		if segment.label == 124:
###		
#from mpl_toolkits.mplot3d import Axes3D
#ax = pylab.axes(projection='3d')
#ax.plot(ap_axis.x, ap_axis.y, ap_axis.z)
#pylab.hold(True)
###ax.plot([self.x[closest_pos], centroid[0]], [self.y[closest_pos], centroid[1]], [self.z[closest_pos], centroid[2]], 'r')
##pylab.axis('equal')
###pylab.show()
##			pdb.set_trace()

		segment.add_feature("angle_with_AP_axis", angle)
