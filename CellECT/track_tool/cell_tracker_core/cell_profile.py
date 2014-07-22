# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import copy
import pdb


from CellECT.track_tool.tissue_selection import select_similar
import CellECT.seg_tool.globals
from CellECT.seg_tool.seg_utils import bounding_box as bounding_box_module
"""
CellProfile and CellProfilesPerTimestamp classes.
Creates a profile for every cell at a given segmentation (time stamp) based on 
the output of CellECT_seg_tool in the segment collection xml file.
Also contains statistics about the cell properties.
"""



class CellProfile(object):

	"""
	Cell profile class, contains information about segments in the
	segment collection xml output from CellECT_seg_tool.
	"""

	def __init__(self, label, nucleus, size, bbx, neighbor_labels, feature_dict):
		self.label = label
		self.nucleus = nucleus
		self.size = size		
		self.neighbor_labels = neighbor_labels
		self.dict_of_features = feature_dict
		self.bounding_box = bbx



class CellProfilesPerTimestamp(object):

	"""
	Contains cell profiles for every cell in the dataset at a given time stamp.
	Also contains statistics of these cells.
	"""

	def __init__(self, time_stamp, input_list_of_cell_profiles):

		""" Cell profile for every cell 
		"""

		self.list_of_cell_profiles = copy.deepcopy(input_list_of_cell_profiles)
		self.time_stamp = time_stamp
		self.feature_histograms = {}
#		self.get_stats(100000)

		self.seg_label_to_cp_list_index = dict((cp.label, index) for index, cp in enumerate(self.list_of_cell_profiles) )		

	def get_stats(self, max_size):

		sizes_list = [x.size for x in self.list_of_cell_profiles]
		hist = np.histogram( sizes_list, 25, (0, max_size) )
		self.size_hist_bins = hist[1]

		self.size_hist_vals = hist[0] / float(np.sum(hist[0]))

		self.size_mean = np.mean(sizes_list)
		self.size_stdev = np.std(sizes_list)



	def get_cells_within_space(self,bbx):
		# return cells whose centroid falls in a space
	
		result = []

		for cp in self.list_of_cell_profiles:
			if cp.nucleus.x > bbx.xmin and cp.nucleus.x < bbx.xmax and \
               cp.nucleus.y > bbx.ymin and cp.nucleus.y < bbx.ymax and \
               cp.nucleus.z > bbx.zmin and cp.nucleus.z < bbx.zmax:
				result.append(cp)

		return result


	def get_cells_fully_within_space(self, bbx):

		# return cells fully contained in the bounding box.

		result = []

		for cp in self.list_of_cell_profiles:
			if cp.bounding_box.xmin > bbx.xmin and cp.bounding_box.xmax < bbx.xmax and \
               cp.bounding_box.ymin > bbx.ymin and cp.bounding_box.ymax < bbx.ymax and \
               cp.bounding_box.zmin > bbx.zmin and cp.bounding_box.zmax < bbx.zmax:
				result.append(cp)

		return result


	def get_bounding_box_of_group(self,cell_labels):
		
			xmin = +1000000
			xmax = -1
			ymin = +1000000
			ymax = -1
			zmin = +1000000
			zmax = -1

			for label in cell_labels:

				cp_index = self.seg_label_to_cp_list_index[label]
				cp = self.list_of_cell_profiles[cp_index]

				bbx = cp.bounding_box
				xmin = min(xmin, bbx.xmin)
				ymin = min(ymin, bbx.ymin)
				zmin = min(zmin, bbx.zmin)

				xmax = max(xmax, bbx.xmax)
				ymax = max(ymax, bbx.ymax)
				zmax = max(zmax, bbx.zmax)

			if xmin !=1000000 and ymin != 1000000 and zmin !=1000000 and xmax != -1 and ymax != -1 and zmax != -1:
				return bounding_box_module.BoundingBox(xmin, xmax, ymin, ymax, zmin, zmax)
			else:
				return None
			

	def get_target_cells_size(self,target_cells):

		stddev_size = 0
		mean_size = 0

		if len(target_cells) ==0:
			return 0,0
	

		if len(target_cells) >1:

			cell_sizes = [cp.size for cp in target_cells]
			mean_size = np.mean(cell_sizes)
			stddev_size = np.std(cell_sizes)
		else:
			stddev_size = self.size_stdev
			mean_size = target_cells[0].size

		print mean_size, stddev_size


		return mean_size, stddev_size

	def get_similar(self,cell_cp_index):

		target_cells = []

		similar_cells_cp_index = set()


		# get cell profile index and cell profiles for the labels of interest
		for index in cell_cp_index:
			target_cells.append(self.list_of_cell_profiles[index])


		target_cell_indices_set = set(cell_cp_index)

		mean_size, stddev_size = self.get_target_cells_size(target_cells)

		# test every cell against the target cells
		for index in self.seg_label_to_cp_list_index.values():

			# if this is a target cell, skip
			if index in target_cell_indices_set:
				continue

			cp = self.list_of_cell_profiles[index]

			# for every target cell
			for cp_target in target_cells:


				if abs(mean_size - cp.size) < stddev_size:

					if index not in similar_cells_cp_index:
						similar_cells_cp_index.add(index)



		return list(similar_cells_cp_index)


	def get_similar_new(self,cell_cp_index):

		# TODO: this doesnt work for shit.


		target_cells = []

		similar_cells_cp_index = set()

		similar_detector = select_similar.TissueSelector()


		features_of_interest = CellECT.seg_tool.globals.specific_features

		training_vect = []
		test_vect = []
		target_cell_indices_set = set(cell_cp_index)

		test_index = []

		# get cell profile index and cell profiles for the labels of interest
		for index in self.seg_label_to_cp_list_index.values():

			features = self.list_of_cell_profiles[index].dict_of_features
			temp_vect = []
			for feat in features_of_interest:
				if features.has_key(feat):
					if isinstance(features[feat], list):
						temp_vect.extend(features[feat])
					else:
						temp_vect.extend([features[feat]])
			if index in target_cell_indices_set:
				training_vect.append(temp_vect)
			else:
				test_vect.append(temp_vect)
				test_index.append(index)

					

		similar_detector.add_training_examples(training_vect)
		similar_detector.train()

		similar_cells_cp_index = []

		import math

		for vect, index in zip(test_vect, test_index):

			val = similar_detector.test_sample(vect)
			if val > 0:
				similar_cells_cp_index.append(index)



		return list(similar_cells_cp_index)




		

