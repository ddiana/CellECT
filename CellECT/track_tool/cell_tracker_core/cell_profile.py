# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import copy


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

	def __init__(self, label, nucleus, size):
		self.label = label
		self.nucleus = nucleus
		self.size = size		




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
		self.get_stats()

		self.seg_label_to_cp_list_index = dict((cp.label, index) for index, cp in enumerate(self.list_of_cell_profiles) )		

	def get_stats(self):

		sizes_list = [x.size for x in self.list_of_cell_profiles]
		hist = np.histogram( sizes_list, 25, (100,10000) )
		self.size_hist_bins = hist[1]
		self.size_hist_vals = hist[0] / float(np.sum(hist[0]))
		self.size_mean = np.mean(sizes_list)
		self.size_stdev = np.std(sizes_list)

		

