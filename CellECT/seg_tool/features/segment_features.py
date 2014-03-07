# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
from scipy import ndimage
import time
import logging
from numpy import histogram
import cv


# Imports from this project
from CellECT.seg_tool.seg_utils import misc
from CellECT.seg_tool.segment_collection import segment_collection as segc
import CellECT.seg_tool.globals
from CellECT.seg_tool.seg_utils import voxel as vx


"""
Functions for different segment properties extraction:
- distance from border to nucleus
- euclidian distance btw two Voxel objects
- average intensity in a mask
- nucleus assignment to a segment (which nucleus generated said segment?)
- ratio of border intensity to interior
- extract features for every segment in the segment collection.
"""


def segment_border_to_nucleus(segment):

	"""
	Return a list of all the distance values from the border to the nucleus.
	"""


	box_bounds = segment.bounding_box

	nucleus = vx.Voxel(segment.nucleus_list[0].x - box_bounds.xmin, segment.nucleus_list[0].y - box_bounds.ymin, segment.nucleus_list[0].z-box_bounds.zmin)

	cropped_mask = np.zeros((box_bounds.xmax - box_bounds.xmin+1, box_bounds.ymax - box_bounds.ymin+1, box_bounds.zmax - box_bounds.zmin+1))

	for voxel in segment.list_of_voxel_tuples:
		cropped_mask[voxel[0] - box_bounds.xmin, voxel[1] - box_bounds.ymin, voxel[2] - box_bounds.zmin] = 1
		
	cropped_mask_eroded = ndimage.morphology.binary_erosion(cropped_mask )
	cropped_mask_border = cropped_mask - cropped_mask_eroded

	[x,y,z] = np.nonzero(cropped_mask_border)

	dist_values = []

	for i in xrange(len(x)):

		dist_values.append( euclidian_distance( vx.Voxel(x[i], y[i], z[i]), nucleus ))


	return dist_values




def euclidian_distance(pixel, nucleus):
	
	"""
	For two voxels/nuclei, compute eucledian distance of the coordinates.
	"""

	zscale = 5
	return np.sqrt((pixel.x - nucleus.x)**2 + (pixel.y - nucleus.y)**2 + (zscale*pixel.z - zscale*nucleus.z)**2)



def avg_intensity(vol, mask):

	"""
	Average intensity in a given mask.
	"""

	sum_intensity = (vol*mask).sum()
	sum_voxels = mask.sum()
	
	return float(sum_intensity) / sum_voxels



def histogram_in_mask(vol, mask):

	new_vol = vol * mask - (1 - mask)	
	bins = [-1]
	num_bins = 20
	bins.extend(255./num_bins * x for x in xrange(num_bins))
	hist = histogram(new_vol, bins = bins)[0]
	hist = hist[1:]
	hist = hist / np.float(hist.sum())
	return hist



def hist_dif(h1, h2):


	sig1 = zip(h1, range(len(h1)))
	a64 = cv.fromarray(np.array(sig1))
	a32 = cv.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
	cv.Convert(a64, a32)

	sig2 = zip(h2, range(len(h2)))
	b64 = cv.fromarray(np.array(sig2))
	b32 = cv.CreateMat(b64.rows, b64.cols, cv.CV_32FC1)
	cv.Convert(b64, b32)

	return cv.CalcEMD2(a32,b32,cv.CV_DIST_L2)

	


def add_nucleus_to_segments(segment_collection, nuclei_collection, label_map):

	"""
	Find the nucleus that generated said segment, assign it to the segment object.
	"""

	# first run through nuclei and pick segment they are in

	segment_collection.update_index_dict()

	for nucleus in nuclei_collection.nuclei_list:

		segment_label = label_map[nucleus.x, nucleus.y, nucleus.z]

		# if it's not on the border, or no data (ground truth map)
		if segment_label > 0:
			# if the segment has been included in this segment collection
			if segment_collection.segment_label_to_list_index_dict.has_key(segment_label):
				segment_index = segment_collection.segment_label_to_list_index_dict[segment_label]

				segment_collection.list_of_segments[segment_index].add_nucleus( nucleus)
		

	# if any segments are left without nuclei, pick the closest nucleus from outside the segment

	for segment in segment_collection.list_of_segments:
		try:
			segment.nucleus_list[0]
		except:
			nucleus = nuclei_collection.find_closest_nucleus_to_segment(segment)
			segment.add_nucleus(nucleus)
			



def segment_border_to_interior_intensity(vol, segment, label_map):

	"""
	Intensity ratio between the segment boundary and the segment interior.
	"""


	box_bounds = segment.bounding_box


	
	cropped_vol = vol[box_bounds.xmin:box_bounds.xmax, box_bounds.ymin:box_bounds.ymax, box_bounds.zmin:box_bounds.zmax]
	
	
	#cropped_mask = np.zeros((box_bounds.xmax - box_bounds.xmin, box_bounds.ymax - box_bounds.ymin, box_bounds.zmax - box_bounds.zmin))
	
	
	cropped_mask = label_map[box_bounds.xmin:box_bounds.xmax, box_bounds.ymin:box_bounds.ymax, box_bounds.zmin:box_bounds.zmax] == segment.label
	
	#for voxel in segment.list_of_voxel_tuples:
	#	cropped_mask[voxel[0] - box_bounds.xmin-1, voxel[1] - box_bounds.ymin-1, voxel[2] - box_bounds.zmin-1] = 1

	cropped_mask_dilated = ndimage.morphology.binary_dilation(cropped_mask )
	cropped_mask_eroded = ndimage.morphology.binary_erosion(cropped_mask )
	
	if cropped_mask_eroded.sum() < 10:
		cropped_mask_eroded = cropped_mask
	
	cropped_mask_border = cropped_mask_dilated - cropped_mask_eroded

	interior_intensity = avg_intensity(cropped_vol,cropped_mask_eroded)	
	border_intensity = avg_intensity(cropped_vol,cropped_mask_border)
	
	if interior_intensity < 0.001:
		interior_intensity = 0.001


	return border_intensity / interior_intensity	




def get_segments_with_features(vol, label_map, set_of_labels, name_of_parent, nuclei_collection):

	"""
	Make segment collection and add features to all the segments.
	"""

	#X, Y = np.meshgrid(range(vol.shape[0]), range(vol.shape[1]))


	list_of_segments  = []

	total = len(set_of_labels)
	counter = 0
	
	message = "Making segment collection of %d segments from %s ..." % (len(set_of_labels), name_of_parent)
	print message
	logging.info(message)

	t1 = time.time()
	segment_collection = segc.SegmentCollection(set_of_labels, label_map, name_of_parent)
	t2 = time.time()
	print "....... %.3f sec                         " %(t2 - t1)
	logging.info ("... %.3f sec" % (t2-t1))

	

	message = "Getting properties for %d segments from %s ..." % (len(set_of_labels), name_of_parent)
	print message
	logging.info(message)
	
	t1 = time.time()

	add_nucleus_to_segments(segment_collection, nuclei_collection, label_map)

	for segment in segment_collection.list_of_segments:
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
			segment.add_feature("size", len(segment.list_of_voxel_tuples))
	
		box_bounds = segment.bounding_box
				
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_intensity"]):

			segment.add_feature("border_to_interior_intensity_ratio", segment_border_to_interior_intensity(vol, segment, label_map))
			segment.add_feature("interior_intensity_hist", histogram_in_mask(vol[box_bounds.xmin:box_bounds.xmax+1, box_bounds.ymin:box_bounds.ymax+1, box_bounds.zmin:box_bounds.zmax+1], segment.mask))
			segment.add_feature("border_intensity_hist", histogram_in_mask(vol[box_bounds.xmin:box_bounds.xmax+1, box_bounds.ymin:box_bounds.ymax+1, box_bounds.zmin:box_bounds.zmax+1], segment.border_mask))
			segment.add_feature("border_to_interior_intensity_hist_dif", hist_dif(segment.feature_dict["border_intensity_hist"], segment.feature_dict["interior_intensity_hist"]))


		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			dist_vector = segment_border_to_nucleus(segment)
			segment.add_feature("border_to_nucleus_distance",dist_vector)
		
			dist_hist = np.histogram(dist_vector,  bins = range(1,100,10) )
			# if the segment is tiny:
			if dist_hist[0].sum() == 0:
				dist_hist = dist_hist[0]
				dist_hist[0] = 1.0
			else:
				dist_hist = dist_hist[0] / float (np.sum(dist_hist[0]))
			segment.add_feature("border_to_nucleus_distance_hist", dist_hist)
			segment.add_feature("border_to_nucleus_distance_mean", sum(dist_vector) / float(len(dist_vector)))
			segment.add_feature("border_to_nucleus_distance_std", np.std(dist_vector))
		
		counter += 1
		misc.print_progress(counter, total)

		#print np.mean(segment.feature_dict["border_to_nucleus_distance"]), segment.nucleus.index
	t2 = time.time()
	print "....... %.3f sec                           " % (t2 - t1)
	logging.info ("... %.3f sec" % (t2-t1))


	return segment_collection


