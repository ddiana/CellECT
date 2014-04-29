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
from collections import namedtuple
from scipy.ndimage.morphology import binary_dilation
import math


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


class DistanceFromMargin(object):

	def __init__(self,ws, x_res, y_res, z_res):

		self.x_res = x_res
		self.y_res = y_res
		self.z_res = z_res
		self.min_res = float(max(x_res, y_res, z_res))
		self.x_scale = x_res / self.min_res
		self.y_scale = y_res / self.min_res
		self.z_scale = z_res / self.min_res
		self.x_step =  self.min_res / x_res
		self.y_step =  self.min_res / y_res
		self.z_step =  self.min_res / z_res
		self.dist = ndimage.distance_transform_edt(ws[::self.x_step, ::self.y_step, ::self.z_step] != 1)


	def get_min_dist_for_segment(self, segment):

		return min((self.dist[self.rescale_coords(coords)] for coords in segment.list_of_voxel_tuples))

	def get_mean_dist_for_segment(self, segment):
		return sum((self.dist[self.rescale_coords(coords)] for coords in segment.list_of_voxel_tuples)) / len(segment.list_of_voxel_tuples)

	def get_max_dist_for_segment(self, segment):

		return max((self.dist[self.rescale_coords(coords)] for coords in segment.list_of_voxel_tuples))

	def rescale_coords(self, coords):
		return (coords[0]*self.x_scale, coords[1]*self.y_scale, coords[2]*self.z_scale)


def segment_inner_point(segment):


	box_bounds = segment.bounding_box
	cropped_mask = segment.mask

	dist = ndimage.distance_transform_edt(cropped_mask)
	max_loc = np.argmax(dist)
	max_loc = np.unravel_index(max_loc, cropped_mask.shape)

	max_loc = (max_loc[0] + box_bounds.xmin, max_loc[1] + box_bounds.ymin, max_loc[2] + box_bounds.zmin)

	return max_loc



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

	for nucleus in nuclei_collection.list_head_nuclei():

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
			

def get_bounding_box_union_two_segments(segment1, segment2):

	CollectiveBoundingBox = namedtuple("CollectiveBoundingBox",["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

	x_min = min(segment1.bounding_box.xmin, segment2.bounding_box.xmin)
	y_min = min(segment1.bounding_box.ymin, segment2.bounding_box.ymin)
	z_min = min(segment1.bounding_box.zmin, segment2.bounding_box.zmin)
	x_max = max(segment1.bounding_box.xmax, segment2.bounding_box.xmax)
	y_max = max(segment1.bounding_box.ymax, segment2.bounding_box.ymax)
	z_max = max(segment1.bounding_box.zmax, segment2.bounding_box.zmax)

	return CollectiveBoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)

	
def get_bounding_box_intersection_two_segments(segment1, segment2):

	CollectiveBoundingBox = namedtuple("CollectiveBoundingBox",["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

	x_min = max(segment1.bounding_box.xmin, segment2.bounding_box.xmin)
	y_min = max(segment1.bounding_box.ymin, segment2.bounding_box.ymin)
	z_min = max(segment1.bounding_box.zmin, segment2.bounding_box.zmin)
	x_max = min(segment1.bounding_box.xmax, segment2.bounding_box.xmax)
	y_max = min(segment1.bounding_box.ymax, segment2.bounding_box.ymax)
	z_max = min(segment1.bounding_box.zmax, segment2.bounding_box.zmax)

	return CollectiveBoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)


def add_neighbor_border_properties( segment1, segment2, vol):

	if segment1.label < segment2.label:
	# only process them once
		return

	bbx = get_bounding_box_intersection_two_segments(segment1, segment2)

	vol_crop = vol[bbx.xmin:bbx.xmax, bbx.ymin:bbx.ymax, bbx.zmin:bbx.zmax]

	CollectiveBoundingBox = namedtuple("CollectiveBoundingBox",["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])

	bbx1 = segment1.bounding_box
	bbx2 = segment2.bounding_box
	
	mask1 = binary_dilation(segment1.border_mask[bbx.xmin-bbx1.xmin : bbx.xmax-bbx1.xmin, bbx.ymin-bbx1.ymin : bbx.ymax-bbx1.ymin, bbx.zmin-bbx1.zmin : bbx.zmax-bbx1.zmin] ,structure=np.ones((3,3,1)))
	mask2 = binary_dilation(segment2.border_mask[bbx.xmin-bbx2.xmin : bbx.xmax-bbx2.xmin, bbx.ymin-bbx2.ymin : bbx.ymax-bbx2.ymin, bbx.zmin-bbx2.zmin : bbx.zmax-bbx2.zmin] ,structure=np.ones((3,3,1)))

	mask = mask1*mask2


	s1 = mask1.sum()
	s2 = mask2.sum()
	s = mask.sum()
	segment1.feature_dict["percent_border_with_neighbor"].append((segment2.label, s/float(s1) ))
	segment2.feature_dict["percent_border_with_neighbor"].append((segment1.label, s/float(s2) ))
	segment1.feature_dict["size_border_with_neighbor"].append((segment1.label, s))
	segment2.feature_dict["size_border_with_neighbor"].append((segment2.label, s))
	segment1.feature_dict["mean_intensity_border_with_neighbor"].append((segment2.label, np.sum(mask*vol_crop)/float(s)))
	segment2.feature_dict["mean_intensity_border_with_neighbor"].append((segment1.label, segment1.feature_dict["mean_intensity_border_with_neighbor"][-1][1]))
	
	score1 = segment1.feature_dict["mean_intensity_border_with_neighbor"][-1][1] * ( 1 - segment1.feature_dict["percent_border_with_neighbor"][-1][1])
	segment1.feature_dict["weighted_merge_score"].append((segment2.label, score1))
	segment2.feature_dict["weighted_merge_score"].append((segment1.label, score1))

#	import pylab

#	pylab.subplot(221)
#	pylab.imshow(vol_crop[:,:,vol_crop.shape[2]/2])
#	pylab.subplot(222)
#	pylab.imshow(mask[:,:,vol_crop.shape[2]/2]*255)
#	pylab.subplot(223)
#	pylab.imshow(mask1[:,:,vol_crop.shape[2]/2]*255)
#	pylab.subplot(224)
#	pylab.imshow(mask2[:,:,vol_crop.shape[2]/2]*255)

#	pylab.show()


#	import pylab

#	pylab.subplot(231)
#	pylab.imshow(mask1[:,:,mask1.shape[2]/2])
#	pylab.subplot(232)
#	pylab.imshow(mask2[:,:,mask2.shape[2]/2])
#	pylab.subplot(233)
#	pylab.imshow(mask[:,:,mask.shape[2]/2])

#	pylab.subplot(234)
#	pylab.imshow(segment1.border_mask[:,:,segment1.border_mask.shape[2]/2])
#	pylab.subplot(235)
#	pylab.imshow(segment2.border_mask[:,:,segment2.border_mask.shape[2]/2])

#	print mask.sum()

#	if mask.sum() ==0:
#		pdb.set_trace()



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


def should_compute_feature(name_of_parent, feature_name):


	if name_of_parent != "test_volume" and not feature_name in CellECT.seg_tool.globals.generic_features:
		return False

	return True


def weighted_mean_from_hist(histogram):

	weights = [x/float(len(histogram)) for x in xrange(len(histogram))]

	mean = sum(p*q for p,q in zip(weights, histogram))

	return mean



def init_neighbor_props_for_segment(segment):

	if not segment.feature_dict.has_key("percent_border_with_neighbor"):
		segment.feature_dict["percent_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("mean_intensity_border_with_neighbor"):
		segment.feature_dict["mean_intensity_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("size_border_with_neighbor"):
		segment.feature_dict["size_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("weighted_merge_score"):
		segment.feature_dict["weighted_merge_score"] = []



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

	dist_metric = None
	x_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["x_res"])
	y_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["y_res"])
	z_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["z_res"])



	if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_dist_from_margin"]) and \
       len(segment_collection.list_of_segments) and \
       segment_collection.list_of_segments[0].name_of_parent == "test_volume":
		dist_metric = DistanceFromMargin(label_map, x_res, y_res, z_res)
	
	t1 = time.time()

	add_nucleus_to_segments(segment_collection, nuclei_collection, label_map)

	for segment in segment_collection.list_of_segments:
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
			if should_compute_feature(segment.name_of_parent, "size"):
				segment.add_feature("size", len(segment.list_of_voxel_tuples))


		if segment.name_of_parent == "test_volume":		


			init_neighbor_props_for_segment(segment)

			for neighbor_label in segment.neighbor_labels:
				seg_idx = segment_collection.segment_label_to_list_index_dict[neighbor_label]
				segment2 = segment_collection.list_of_segments[seg_idx]
				init_neighbor_props_for_segment(segment2)
				add_neighbor_border_properties(segment,segment2, vol )	

		box_bounds = segment.bounding_box
				
		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_intensity"]):

			if should_compute_feature(segment.name_of_parent, "border_to_interior_intensity_ratio"):
				segment.add_feature("border_to_interior_intensity_ratio", segment_border_to_interior_intensity(vol, segment, label_map))
			if should_compute_feature(segment.name_of_parent, "interior_intensity_hist"):
				segment.add_feature("interior_intensity_hist", histogram_in_mask(vol[box_bounds.xmin:box_bounds.xmax+1, box_bounds.ymin:box_bounds.ymax+1, box_bounds.zmin:box_bounds.zmax+1], segment.mask))
			if should_compute_feature(segment.name_of_parent, "interior_weighted_intensity_mean"):
				segment.add_feature("interior_weighted_intensity_mean", weighted_mean_from_hist(segment.feature_dict["interior_intensity_hist"]))
			if should_compute_feature(segment.name_of_parent, "border_intensity_hist"):
				segment.add_feature("border_intensity_hist", histogram_in_mask(vol[box_bounds.xmin:box_bounds.xmax+1, box_bounds.ymin:box_bounds.ymax+1, box_bounds.zmin:box_bounds.zmax+1], segment.border_mask))
			if should_compute_feature(segment.name_of_parent, "border_to_interior_intensity_hist_dif"):
				segment.add_feature("border_to_interior_intensity_hist_dif", hist_dif(segment.feature_dict["border_intensity_hist"], segment.feature_dict["interior_intensity_hist"]))


		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_dist_from_margin"]):

			if should_compute_feature(segment.name_of_parent, "distance_from_margin"):
				segment.add_feature("min_distance_from_margin", dist_metric.get_min_dist_for_segment(segment))
				segment.add_feature("mean_distance_from_margin", dist_metric.get_mean_dist_for_segment(segment))
				segment.add_feature("max_distance_from_margin", dist_metric.get_max_dist_for_segment(segment))

		if should_compute_feature(segment.name_of_parent, "inner_point"):
				segment.add_feature("inner_point", segment_inner_point(segment))
			

		if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_border_distance"]):
			dist_vector = segment_border_to_nucleus(segment)
			if should_compute_feature(segment.name_of_parent, "border_to_nucleus_distance"):
				segment.add_feature("border_to_nucleus_distance",dist_vector)
		
			dist_hist = np.histogram(dist_vector,  bins = range(1,100,10) )
			# if the segment is tiny:
			if dist_hist[0].sum() == 0:
				dist_hist = dist_hist[0]
				dist_hist[0] = 1.0
			else:
				dist_hist = dist_hist[0] / float (np.sum(dist_hist[0]))
			if should_compute_feature(segment.name_of_parent, "border_to_nucleus_distance_hist"):
				segment.add_feature("border_to_nucleus_distance_hist", dist_hist)
			if should_compute_feature(segment.name_of_parent, "border_to_nucleus_distance_mean"):
				segment.add_feature("border_to_nucleus_distance_mean", sum(dist_vector) / float(len(dist_vector)))
			if should_compute_feature(segment.name_of_parent, "border_to_nucleus_distance_std"):
				segment.add_feature("border_to_nucleus_distance_std", np.std(dist_vector))
		
		counter += 1
		misc.print_progress(counter, total)

		#print np.mean(segment.feature_dict["border_to_nucleus_distance"]), segment.nucleus.index
	t2 = time.time()
	print "....... %.3f sec                           " % (t2 - t1)
	logging.info ("... %.3f sec" % (t2-t1))


	return segment_collection


