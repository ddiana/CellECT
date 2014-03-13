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


# Imports from this project
from CellECT.seg_tool.seg_utils import bounding_box as bbx


"""
Collection of segents and tools to manipuate them.
"""

class SegmentCollection(object):
	
	"Collection of segments and tools to manipulate them."

	def __init__ (self, set_of_labels, label_map, name_of_parent):

		self.name_of_parent = name_of_parent

		self.list_of_segments = []
		
		self.add_segments_to_collection(label_map, set_of_labels, name_of_parent)



	def make_contours_for_all_segments(self, label_map):

		for segment in self.list_of_segments:
			segment.make_segment_contours(label_map)

	def add_segments_to_collection(self, label_map, set_of_labels, name_of_parent):
	
		"""Given a label map, and a list of labels of interest, 
		add those segments to the segment collection."""		

		# prepares list of segments voxels for every segment

		label_map = label_map.astype("int32")
		
		reverse_index = {label: [] for label in set_of_labels}

		it = np.nditer(label_map, flags=['multi_index'])
		
		while not it.finished:
			label = int(it[0])
			if label in set_of_labels:
				reverse_index[label].append(it.multi_index)

			it.iternext()		
		
		for label in set_of_labels:
			self.list_of_segments.append(Segment(int(label), reverse_index[label], name_of_parent, label_map.shape, label_map))	

		
	def add_segment_using_mask(self, label_map, label, name_of_parent):
	
		"Add a segment with a specifit label."

		mask = (label_map == label)
		voxels = zip(*dld_nonzero3d(mask))
		
		
		self.list_of_segments.append(Segment(label, voxels, name_of_parent, label_map.shape, label_map))
		

	def update_index_dict(self):

		self.segment_label_to_list_index_dict = dict((segment.label, index) for index, segment in enumerate(self.list_of_segments))



	def get_feature_values_in_list(self, feature_name):
		
		feat_list = [segment.feature_dict[feature_name] for segment in self.list_of_segments]
		return feat_list
		



class Segment(object):

	"Segment class, includes segment features, nucleus, bounding box, etc."

	def __init__ (self, label, voxel_location_tuples, name_of_parent, max_shape, label_map):

		self.label = label
		self.list_of_voxel_tuples = voxel_location_tuples
		self.name_of_parent = name_of_parent
		self.feature_dict = {}
		self.nucleus_list = []
		self.bounding_box = self.get_boundaries()
		self.bounding_box.extend_by (5, max_shape)
		self.contour_polygons_list = []
		
		self.mask = None
		self.set_mask()
		self.border_mask = None
		self.set_border_mask()
		self.neighbor_labels = set()
		self.get_neighbors(label_map)


	def set_mask(self, ):

		self.mask = np.zeros((self.bounding_box.xmax - self.bounding_box.xmin+1, self.bounding_box.ymax - self.bounding_box.ymin+1, self.bounding_box.zmax - self.bounding_box.zmin+1))
		for i,j,k in self.list_of_voxel_tuples:
			self.mask[i-self.bounding_box.xmin, j - self.bounding_box.ymin, k - self.bounding_box.zmin] = 1

	def set_border_mask(self):

		self.border_mask = binary_dilation(self.mask, structure=np.ones((3,3,1))) - self.mask

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
			contour_output = cv2.findContours(cropped_mask[:,:,z].astype('uint8'),cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE,offset = (self.bounding_box.ymin,self.bounding_box.xmin))
			if contour_output[0]:
				polygon = [(contour_output[0][0][i][0][1], contour_output[0][0][i][0][0], z + self.bounding_box.zmin) for i in xrange(len(contour_output[0][0]))]
				self.contour_polygons_list.append(polygon)
	

		


	def add_feature(self,feat_name, feat_value):
	
		"Add a feature to the feature dictionary of the segment."

		# TODO check if exists
		self.feature_dict[feat_name] = feat_value		


	def get_boundaries(self):

		"Get the boundaries of this segment. Useful to crop a bounding box around it when needed."

		#x,y,z = zip(*self.list_of_voxel_tuples)

		(xmin, ymin, zmin) = reduce(lambda a,b: (min(a[0],b[0]), min(a[1], b[1]), min(a[2], b[2])), self.list_of_voxel_tuples)		
		(xmax, ymax, zmax) = reduce(lambda a,b: (max(a[0],b[0]), max(a[1], b[1]), max(a[2], b[2])), self.list_of_voxel_tuples)		


		box_bounds = bbx.BoundingBox( xmin, xmax, ymin, ymax, zmin, zmax)


		return box_bounds	

	def add_nucleus(self, nucleus):
		self.nucleus_list.append(nucleus)



