# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import time

# Imports from this project
from seg_utils import bounding_box as bbx


"""
Collection of segents and tools to manipuate them.
"""

class SegmentCollection(object):
	
	"Collection of segments and tools to manipulate them."

	def __init__ (self, set_of_labels, label_map, name_of_parent):

		self.name_of_parent = name_of_parent

		self.list_of_segments = []
		
		self.add_segments_to_collection(label_map, set_of_labels, name_of_parent)

	
	def add_segments_to_collection(self, label_map, set_of_labels, name_of_parent):
	
		"""Given a label map, and a list of labels of interest, 
		add those segments to the segment collection."""		

		# prepares list of segments voxels for every segment

		label_map = label_map.astype("int32")
		
		t1 = time.time()

		reverse_index = {label: [] for label in set_of_labels}

		it = np.nditer(label_map, flags=['multi_index'])
		
		while not it.finished:
			label = int(it[0])
			if label in set_of_labels:
				reverse_index[label].append(it.multi_index)

			it.iternext()
		

		print ".......",  time.time() - t1, "sec          "
		
		
		for label in set_of_labels:
			self.list_of_segments.append(Segment(int(label), reverse_index[label], name_of_parent))	

		
	def add_segment_using_mask(self, label_map, label, name_of_parent):
	
		"Add a segment with a specifit label."

		mask = (label_map == label)
		voxels = zip(*dld_nonzero3d(mask))
		
		
		self.list_of_segments.append(Segment(label, voxels, name_of_parent))
		

	def update_index_dict(self):

		self.segment_label_to_list_index_dict = dict((segment.label, index) for index, segment in enumerate(self.list_of_segments))



	def get_feature_values_in_list(self, feature_name):
		
		feat_list = [segment.feature_dict[feature_name] for segment in self.list_of_segments]
		return feat_list
		



class Segment(object):

	"Segment class, includes segment features, nucleus, bounding box, etc."

	def __init__ (self, label, voxel_location_tuples, name_of_parent):

		self.label = label
		self.list_of_voxel_tuples = voxel_location_tuples
		self.name_of_parent = name_of_parent
		self.feature_dict = {}
		self.nucleus_list = []
		self.bounding_box = self.get_boundaries()
		
	

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



