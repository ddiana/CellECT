# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import copy
import time
import logging

# Imports from this project
from CellECT.seg_tool.seg_utils import bounding_box as bbx

"""
Segments that come from seeds and tools to manipulate them.
"""

class SeedSegmentCollection(object):

	"Collection of segments which come from seeds."

	def __init__(self):

		self.list_of_seed_segments = []
		self.index_to_list_index_dict = {}


	def update_seed_segment_collection(self,seed_segment_collection, label_map, seed_collection):

		"Making seed segment collection from seeds and label map info."

		# prepares list of segments voxels for every segment
		

		label_map = label_map.astype("int32")
		
		t1 = time.time()

		reverse_index = {int(label_map[seed.x, seed.y, seed.z]): [] for seed in seed_collection.list_of_seeds}

		message = "Making seed-segment collection of %d segments from seeds..." % len (reverse_index)
		print message
		logging.info(message)

		it = np.nditer(label_map, flags=['multi_index'])
		labels_of_interest = set(reverse_index.keys())
		while not it.finished:
			label = int(it[0])
			if label in labels_of_interest:
				reverse_index[label].append(it.multi_index)
			it.iternext()

		for seed in seed_collection.list_of_seeds:
			label = label_map[seed.x, seed.y, seed.z]
			self.list_of_seed_segments.append(SeedSegment(seed, reverse_index[label]))	


		self.seed_index_to_seed_segment_list_index_dict = dict(( seed_segment.seed.index, index) for index, seed_segment in enumerate(self.list_of_seed_segments))
	
				
		t2 = time.time()
		print "....... %.3f sec              " % (t2-t1)
		logging.info("... %.3f sec" % (t2-t1))



class SeedSegment(object):

	"Seed segment class. Contains list of voxel, associated seed and bounding box."

	def __init__ (self, seed, list_of_voxel_tuples):
		self.seed = copy.deepcopy(seed)
		self.list_of_voxel_tuples = copy.deepcopy(list_of_voxel_tuples)
		self.bounding_box = self.get_boundaries()


	def get_boundaries(self):

		#x,y,z = zip(*self.list_of_voxel_tuples)

		(xmin, ymin, zmin) = reduce(lambda a,b: (min(a[0],b[0]), min(a[1], b[1]), min(a[2], b[2])), self.list_of_voxel_tuples)		
		(xmax, ymax, zmax) = reduce(lambda a,b: (max(a[0],b[0]), max(a[1], b[1]), max(a[2], b[2])), self.list_of_voxel_tuples)		


		box_bounds = bbx.BoundingBox( xmin, xmax, ymin, ymax, zmin, zmax)

		return box_bounds	
	
