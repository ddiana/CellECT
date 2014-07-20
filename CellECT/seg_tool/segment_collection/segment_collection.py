# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import time
import cv
import cv2
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion
import pylab
from collections import namedtuple
import copy

# Imports from this project
from CellECT.seg_tool.seg_utils import bounding_box as bbx


"""
Collection of segents and tools to manipuate them.
"""

class SegmentCollection(object):
	
	"Collection of segments and tools to manipulate them."

	def __init__ (self, set_of_labels, label_map, name_of_parent, xres, yres, zres):

		self.name_of_parent = name_of_parent

		self.list_of_segments = []
		self.xres = xres
		self.yres = yres
		self.zres = zres

		if len(set_of_labels)>0:
			self.add_segments_to_collection(label_map, set_of_labels, name_of_parent)


		self.update_index_dict



	def make_contours_for_all_segments(self, label_map):

		for segment in self.list_of_segments:
			segment.make_segment_contours(label_map)



	def add_segments_to_collection(self, label_map, set_of_labels, name_of_parent):
	
		"""Given a label map, and a list of labels of interest, 
		add those segments to the segment collection."""		

		# prepares list of segments voxels for every segment



		label_map = label_map.astype("int32")
		

		# [ xmin, xmax, ymin, ymax, zmin, zmax]
		reverse_index = {label: [100000000, -1, 10000000, -1, 100000000, -1] for label in set_of_labels}


		b_first, b_last = self.get_extremity_values(label_map)

#		it = np.nditer(label_map, flags=['multi_index'])
#		
#		pdb.set_trace()
#		print "... Locating segments in label map."		
#		t = time.time()	
#		while not it.finished:
#			label = int(it[0])
#			if label in set_of_labels:
#				#pdb.set_trace()
#				#reverse_index[label].append(it.multi_index)
#				vals = reverse_index[label]

#				reverse_index[label] = [min(vals[0], it.multi_index[0]),  max(vals[1], it.multi_index[0]) , min(vals[2], it.multi_index[1]),  max(vals[3], it.multi_index[1]), min(vals[4], it.multi_index[2]),  max(vals[5], it.multi_index[2])]
#			it.iternext()		
#		print "...... %.3f sec                           " % ( time.time() - t)
#		
#		pdb.set_trace()

		t = time.time()
		for label in set_of_labels:
			extremities = [b_first[0][label], b_last[0][label], b_first[1][label], b_last[1][label], b_first[2][label], b_last[2][label]]
			self.list_of_segments.append(Segment(int(label), extremities, name_of_parent, label_map.shape, label_map, self.xres, self.yres, self.zres))	

		#print "...... %.3f sec                           " % ( time.time() - t)

		

	def get_extremity_values(self, label_map):

#		t= time.time()


		labmax = label_map.max()
		b_first = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')
		b_last = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')

		# run through all dimensions making 2D slices and marking all existing labels to b
		for dim in range(3):
		# create a generic slice object to make the slices
			sl = [slice(None), slice(None), slice(None)]

			bf = b_first[dim]
			bl = b_last[dim]

			# go through all slices in this dimension
			for k in range(label_map.shape[dim]):
				# create the slice object
				sl[dim] = k
				# update the last "seen" vector
				bl[label_map[sl].flatten()] = k

				# if we have smaller values in "last" than in "first", update
				bf[:] = np.clip(bf, None, bl)

#		print time.time() - t

		return b_first, b_last
		


	def add_segment_using_mask(self, label_map, label, name_of_parent):
	
		"Add a segment with a specifit label."

		mask = (label_map == label)
		voxels = zip(*dld_nonzero3d(mask))
		
		
		self.list_of_segments.append(Segment(label, voxels, name_of_parent, label_map.shape, label_map, self.xres, self.yres, self.zres))
		

	def update_index_dict(self):

		self.segment_label_to_list_index_dict = dict((segment.label, index) for index, segment in enumerate(self.list_of_segments))



	def get_feature_values_in_list(self, feature_name):
		
		feat_list = [segment.feature_dict[feature_name] for segment in self.list_of_segments]
		return feat_list
		



class Segment(object):

	"Segment class, includes segment features, nucleus, bounding box, etc."

	def __init__ (self, label, extremities, name_of_parent, max_shape, label_map, xres, yres, zres):


		self.label = label
		#self.list_of_voxel_tuples = voxel_location_tuples
		self.name_of_parent = name_of_parent
		self.feature_dict = {}
		self.nucleus_list = []
		self.bounding_box = self.get_boundaries(extremities)
		self.bounding_box.extend_by (5, max_shape)
		self.contour_polygons_list = []
		self.xres = xres
		self.yres = yres
		self.zres = zres
		
		self.mask = None
		self.set_mask(label_map)
		self.border_mask = None
		self.set_border_mask()
		self.neighbor_labels = set()
		self.get_neighbors(label_map)


	def set_mask(self, label_map):

		#self.mask = np.zeros((self.bounding_box.xmax - self.bounding_box.xmin+1, self.bounding_box.ymax - self.bounding_box.ymin+1, self.bounding_box.zmax - self.bounding_box.zmin+1))
		#for i,j,k in self.list_of_voxel_tuples:
		#	self.mask[i-self.bounding_box.xmin, j - self.bounding_box.ymin, k - self.bounding_box.zmin] = 1

		self.mask = label_map[self.bounding_box.xmin : self.bounding_box.xmax+1, self.bounding_box.ymin : self.bounding_box.ymax+1, self.bounding_box.zmin : self.bounding_box.zmax+1] == self.label
		self.mask = self.mask.astype("uint8")

	def _get_border_pixels_from_mask(self, mask_in):



		mask = np.zeros((mask_in.shape[0]+2 , mask_in.shape[1] + 2, mask_in.shape[2]+2))
		mask[1:-1,1:-1, 1:-1] = mask_in

		dilated = binary_dilation (mask)


		old_x, old_y, old_z = np.nonzero(dilated - mask)
			
		old_x = old_x - 1
		old_y = old_y - 1
		old_z = old_z - 1

		new_x = np.clip(old_x, 0, mask_in.shape[0]-1)
		new_y = np.clip(old_y, 0, mask_in.shape[1]-1)
		new_z = np.clip(old_z, 0, mask_in.shape[2]-1)


		adjusted_boundary = zip(new_x, new_y, new_z)


		return adjusted_boundary


	def set_border_mask(self):

		""""
		BORDER_COORDS: with ref to crop (0,0,0), counts pixels
		ISOTROPIC_BORDER_COORDS: with ref to crop (0,0,0), counts pixels in the best resolution
		BORDER_COORDS_BY_RES: with ref to global (0,0,0), points from border_coords, but in microns
		"""
		

		try:
			self.border_coords = self._get_border_pixels_from_mask(self.mask)

			self.border_mask = np.zeros(self.mask.shape)

			for x in self.border_coords:
				self.border_mask[x] = 1


	#	
	#		from mpl_toolkits.mplot3d import Axes3D
	#		fig = pylab.figure()
	#		ax = fig.add_subplot(111,projection='3d')
	#		to_plot = zip(*self.border_coords)
	#		pylab.plot(to_plot[0], to_plot[1], to_plot[2], '*b')


			xres = self.xres
			yres = self.yres
			zres = self.zres


			x_coords, y_coords, z_coords = zip(*self.border_coords)
			offset_x = self.bounding_box.xmin
			offset_y = self.bounding_box.ymin
			offset_z = self.bounding_box.zmin

			self.border_coords_by_res = zip((np.array(x_coords) + offset_x)*xres, (np.array(y_coords) + offset_y)*yres, (np.array(z_coords) + offset_z)*zres)


			max_res = min([xres, yres, zres])

			x_scale = xres /max_res 
			y_scale = yres /max_res 
			z_scale = zres/ max_res

			temp_mat = None
			if x_scale > 1:
				temp_mat = np.repeat(self.mask, x_scale, 0)
			if y_scale > 1:
				temp_mat = np.repeat(self.mask, y_scale, 1)
			if z_scale > 1:
				temp_mat = np.repeat(self.mask, z_scale, 2)

			boundary = None
			if temp_mat is None:
				self.isotropic_border_coords = self.border_coords
			else:

			
				boundary = self._get_border_pixels_from_mask(temp_mat)

				#box_offset = (self.bounding_box.xmin , self.bounding_box.ymin, self.bounding_box.zmin )

				#new_x, new_y, new_z =  np.array(zip(*boundary))

				self.isotropic_border_coords = boundary  # =zip(new_x + box_offset[0] * x_scale, new_y + box_offset[1] * y_scale, new_z + box_offset[2] * z_scale)

		except:
			pdb.set_trace()
#		


#		fig1 = pylab.figure()
#		ax = fig1.add_subplot(111,projection='3d')
#		to_plot = zip(*boundary)
#		pylab.plot(to_plot[0], to_plot[1], to_plot[2], '*r')




#		fig2 = pylab.figure()
#		ax = fig2.add_subplot(111,projection='3d')
#		to_plot = zip(*self.boundary_coords_by_res)
#		pylab.plot(to_plot[0], to_plot[1], to_plot[2], '*g')
#		pylab.show()



##	

#		import pylab

#		pylab.subplot(211)
#		z = self.mask.shape[2] /2
#		pylab.imshow(self.mask[:,:,z])
#		pylab.subplot(212)
#		pylab.imshow(self.border_mask[:,:,z])
#		pylab.show()

	def get_neighbors(self, label_map):

		cropped_map = label_map[self.bounding_box.xmin : self.bounding_box.xmax+1, self.bounding_box.ymin : self.bounding_box.ymax +1, self.bounding_box.zmin:self.bounding_box.zmax +1]
		neighbors = np.unique( binary_dilation(self.border_mask).astype("int")* cropped_map)
		for x in neighbors:
			if not x in set((0,1,self.label)):
				self.neighbor_labels.add(x)

		

	

	def make_segment_contours(self, label_map):

		
		"Add the polygon contours for every segment as a list of points"

		cropped_mask = self.mask #label_map[self.bounding_box.xmin : self.bounding_box.xmax, self.bounding_box.ymin : self.bounding_box.ymax, self.bounding_box.zmin:self.bounding_box.zmax] == self.label

		for z in xrange(cropped_mask.shape[2]):
			plane = copy.deepcopy(cropped_mask[:,:,z]).astype('uint8')
			offset = (self.bounding_box.ymin,self.bounding_box.xmin)

			contour_output = cv2.findContours(plane.copy() , cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE, offset = offset)
			
			for idx in xrange(len( contour_output[0])):
				
				polygon = [(contour_output[0][idx][i][0][1], contour_output[0][idx][i][0][0], z + self.bounding_box.zmin) for i in xrange(len(contour_output[0][idx]))]
				self.contour_polygons_list.append(polygon)




	def get_mid_slice(self,):

		try:
			mid_slice = self.mid_slice

		except:

			pts = np.array(self.border_coords)
			self.mid_slice_z = np.floor(np.mean(pts,0)[2])
			self.mid_slice = self.mask[:,:,self.mid_slice_z]

#			if self.mid_slice.sum() == 0:
#				print "EMPTY MID SLICE"
#				pdb.set_trace()
			

#			cropped_mask = self.mask

#			z_list = np.unique(np.nonzero(cropped_mask)[2])
#			z = z_list[len(z_list)/2]
#	
#			mid_slice = cropped_mask[:,:,z]

#			mid_slice = binary_dilation (binary_erosion(mid_slice))		
#			if not mid_slice.any():
#				mid_slice = cropped_mask[:,:,z]

#			self.mid_slice = mid_slice
#			self.mid_slice_z = z

		return self.mid_slice


	def get_mid_slice_contour(self):


		try:
			mid_slice_contour = self.mid_slice_contour

		except:

			mid_slice = copy.deepcopy(self.get_mid_slice())

			contour_output = cv2.findContours(mid_slice.astype('uint8').copy(),cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE,offset = (self.bounding_box.ymin,self.bounding_box.xmin))

			#idx = np.argmax([len(x) for x in contour_output[0]])

			idx = 0
			self.mid_slice_contour = [(contour_output[0][idx][i][0][1], contour_output[0][idx][i][0][0]) for i in xrange(len(contour_output[0][idx]))]
	
		return self.mid_slice_contour
	


	def add_feature(self,feat_name, feat_value):
	
		"Add a feature to the feature dictionary of the segment."

		# TODO check if exists
		self.feature_dict[feat_name] = feat_value		


	def get_boundaries(self, extremities):

		"Get the boundaries of this segment. Useful to crop a bounding box around it when needed."

		#x,y,z = zip(*self.list_of_voxel_tuples)

		#(xmin, ymin, zmin) = reduce(lambda a,b: (min(a[0],b[0]), min(a[1], b[1]), min(a[2], b[2])), self.list_of_voxel_tuples)		
		#(xmax, ymax, zmax) = reduce(lambda a,b: (max(a[0],b[0]), max(a[1], b[1]), max(a[2], b[2])), self.list_of_voxel_tuples)		


		xmin = extremities[0]
		xmax = extremities[1]
		ymin = extremities[2]
		ymax = extremities[3]
		zmin = extremities[4]
		zmax = extremities[5]

		box_bounds = bbx.BoundingBox( xmin, xmax, ymin, ymax, zmin, zmax)


		return box_bounds	

	def add_nucleus(self, nucleus):
		self.nucleus_list.append(nucleus)

	



				



