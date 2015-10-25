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
import cv2
from sklearn.decomposition import PCA
import pylab

from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

# Imports from this project
from CellECT.seg_tool.seg_utils import misc
from CellECT.seg_tool.segment_collection import segment_collection as segc
import CellECT.seg_tool.globals
from CellECT.seg_tool.seg_utils import voxel as vx
from CellECT.seg_tool.features import APaxis


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
		self.x_step =  np.floor(1/self.x_scale)
		self.y_step =  np.floor(1/self.y_scale)
		self.z_step =  np.floor(1/self.z_scale)
		self.dist = ndimage.distance_transform_edt(ws[::self.x_step, ::self.y_step, ::self.z_step] != 1)


		

#	def get_min_dist_for_segment(self, segment):
#		
#		# min dist is on the boundary, using boundary pixels.

#		dists = []
#		for i in xrange(len(segment.contour_polygons_list)):
#			dists.extend([self.dist[self.rescale_coords(coords)] for coords in segment.contour_polygons_list[i]])
#		
#		return min(dists)

#	def get_mean_dist_for_segment(self, segment):
#		return sum((self.dist[self.rescale_coords(coords)] for coords in segment.list_of_voxel_tuples)) / len(segment.list_of_voxel_tuples)

	def get_centroid_dist(self, segment):

		try:
			return self.dist[self.rescale_coords(segment.feature_dict["centroid"])] * max([self.x_res, self.y_res, self.z_res])
		except:
			pdb.set_trace()

#	def get_max_dist_for_segment(self, segment):

#		return max((self.dist[self.rescale_coords(coords)] for coords in segment.list_of_voxel_tuples))

	def rescale_coords(self, coords):
		return (int(np.floor(coords[0]/self.x_step)), int(np.floor(coords[1]/self.y_step)), int(np.floor(coords[2]/self.z_step)))


def segment_inner_point(segment):


	box_bounds = segment.bounding_box
	cropped_mask = segment.mask

	dist = ndimage.distance_transform_edt(cropped_mask)
	max_loc = np.argmax(dist)
	max_loc = np.unravel_index(max_loc, cropped_mask.shape)

	max_loc = (max_loc[0] + box_bounds.xmin, max_loc[1] + box_bounds.ymin, max_loc[2] + box_bounds.zmin)

	return max_loc



#def segment_border_to_nucleus(segment):

#	# DO NOT USE THIS. BETTER VERSION AVAILABLE.

#	"""
#	Return a list of all the distance values from the border to the nucleus.
#	"""


#	box_bounds = segment.bounding_box

#	nucleus = vx.Voxel(segment.nucleus_list[0].x - box_bounds.xmin, segment.nucleus_list[0].y - box_bounds.ymin, segment.nucleus_list[0].z-box_bounds.zmin)

#	cropped_mask = np.zeros((box_bounds.xmax - box_bounds.xmin+1, box_bounds.ymax - box_bounds.ymin+1, box_bounds.zmax - box_bounds.zmin+1))

#	for voxel in segment.list_of_voxel_tuples:
#		cropped_mask[voxel[0] - box_bounds.xmin, voxel[1] - box_bounds.ymin, voxel[2] - box_bounds.zmin] = 1
#		
#	cropped_mask_eroded = ndimage.morphology.binary_erosion(cropped_mask )
#	cropped_mask_border = cropped_mask - cropped_mask_eroded

#	[x,y,z] = np.nonzero(cropped_mask_border)

#	dist_values = []

#	for i in xrange(len(x)):

#		dist_values.append( euclidian_distance( vx.Voxel(x[i], y[i], z[i]), nucleus ))


#	return dist_values




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
	max_val = 256  # way over 255
	bins.extend([float(max_val)/(num_bins) * x for x in xrange(num_bins+1)])
	hist = histogram(new_vol, bins = bins)[0]
	hist = hist[1:]
	hist = hist / np.float(hist.sum())
	return hist

def get_pca_properties(segment):


	X = np.array(zip(*segment.border_coords_by_res)).T
	pca = PCA(n_components=3)
	result = pca.fit(X)

	segment.add_feature("pca_eigenvectors", result.components_)
	segment.add_feature("pca_eigenvalues", result.explained_variance_ratio_)





def get_min_oriented_bounding_box_properties(segment):


	x_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["x_res"])
	y_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["y_res"])
	z_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["z_res"])


	pts_pca0 = []
	pts_pca1 = []
	pts_pca2 = []

	pca_0 = segment.feature_dict["pca_eigenvectors"][0]
	pca_1 = segment.feature_dict["pca_eigenvectors"][1]
	pca_2 = segment.feature_dict["pca_eigenvectors"][2]

#	for pt in segment.:

#		new_pt = np.array([(pt[0] - xc)* x_res, (pt[1] - yc) * y_res, (pt[2] - zc) * z_res])

#		pts_pca0.append( np.dot (pca_0, new_pt) )
#		pts_pca1.append( np.dot (pca_1, new_pt) )
#		pts_pca2.append( np.dot (pca_2, new_pt) )

	input_pts = np.array(segment.border_coords_by_res)
	means = np.mean(input_pts)
	input_pts = np.subtract(input_pts, means)

	pts_pca0 = np.dot(input_pts, pca_0) 		#[np.dot (pca_0, x) for x in segment.border_coords_by_res]
	pts_pca1 = np.dot(input_pts, pca_1)	#[np.dot (pca_1, x) for x in segment.border_coords_by_res]
	pts_pca2 = np.dot(input_pts, pca_2)		#[np.dot (pca_2, x) for x in segment.border_coords_by_res]


	bbx_pca0 = max(pts_pca0) - min(pts_pca0)
	bbx_pca1 = max(pts_pca1) - min(pts_pca1)
	bbx_pca2 = max(pts_pca2) - min(pts_pca2)

	dist_pca0 = np.abs(pts_pca0)
	dist_pca1 = np.abs(pts_pca1)
	dist_pca2 = np.abs(pts_pca2)
	
#	centroid_distances = list(dist_pca0)
#	centroid_distances.extend(list(dist_pca1))
#	centroid_distances.extend(list(dist_pca2))

	sphere_radius = segment.feature_dict["distance_to_border_scale_factor"] * min([x_res, y_res, z_res])

	segment_volume = segment.feature_dict["size"] * x_res * y_res * z_res

	segment.add_feature("volume_by_res", segment_volume)

	segment.add_feature("surface_area_by_res", len(segment.isotropic_border_coords) * min([x_res, y_res, z_res])**2)

	# spherecity

	segment.add_feature("minimum_enclosing_sphere_radius_by_res", sphere_radius)

	sphere_vol_ratio =  segment_volume  / (4/3. * np.pi * sphere_radius ** 3 )
	segment.add_feature("volume_by_res_to_enclosing_sphere_vol_ratio", sphere_vol_ratio)


	equivalent_sphere_radius = (3 * segment_volume / 4. / np.pi)**(1./3)
	equivalent_sphere_surface_area = 4 * np.pi * equivalent_sphere_radius**2

	sphericity = segment.feature_dict["surface_area_by_res"] /equivalent_sphere_surface_area 

	segment.add_feature("sphericity", sphericity)

	# oriented box

	segment.add_feature("minimum_oriented_bbx_side_length_by_res",  [bbx_pca0, bbx_pca1, bbx_pca2] )
	segment.add_feature("minimum_oriented_bbx_volume", bbx_pca0 * bbx_pca1 * bbx_pca2)


	ordered = sorted(segment.feature_dict["minimum_oriented_bbx_side_length_by_res"])

	# elongation and flatness

	segment.add_feature("elongation" ,  float(ordered[2] ) / ordered[1])
	segment.add_feature("flatness" , float(ordered[1] ) / ordered[0] )

	# squareness

	segment.add_feature("vol_by_res_to_enclosing_box_vol_ratio",  float(segment.feature_dict["volume_by_res"])/ (bbx_pca0 * bbx_pca1 * bbx_pca2))

	# cylinder

	segment.add_feature("cylinder_radius_height_pca0", (max(dist_pca0), bbx_pca0))
	segment.add_feature("cylinder_radius_height_pca1", (max(dist_pca1), bbx_pca1))
	segment.add_feature("cylinder_radius_height_pca2", (max(dist_pca2), bbx_pca2))

	cylinder_vol = lambda radius, height: np.pi * radius**2 * height
	cylinder_area = lambda radius, height: 2* np.pi * radius * height + 2 * np.pi * radius**2

	segment.add_feature("vol_cylinder_pca0", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca0"]))
	segment.add_feature("vol_cylinder_pca1", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca1"]))
	segment.add_feature("vol_cylinder_pca2", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca2"]))

	segment.add_feature("surf_area_cylinder_pca0", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca0"]))
	segment.add_feature("surf_area_cylinder_pca1", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca1"]))
	segment.add_feature("surf_area_cylinder_pca2", cylinder_vol(*segment.feature_dict["cylinder_radius_height_pca2"]))

	min_vol_pca = 0
	min_vol = min([segment.feature_dict["vol_cylinder_pca0"], segment.feature_dict["vol_cylinder_pca1"],segment.feature_dict["vol_cylinder_pca2"]])

	if segment.feature_dict["vol_cylinder_pca1"] <  segment.feature_dict["vol_cylinder_pca0"] and  segment.feature_dict["vol_cylinder_pca1"] <  segment.feature_dict["vol_cylinder_pca2"] :
		min_vol_pca = 1
	
	if segment.feature_dict["vol_cylinder_pca2"] <  segment.feature_dict["vol_cylinder_pca0"] and  segment.feature_dict["vol_cylinder_pca2"] <  segment.feature_dict["vol_cylinder_pca1"] :
		min_vol_pca = 2


	segment.add_feature("minimum_vol_cylinder_radius_height", min_vol )
	segment.add_feature("minimum_vol_cylinder_pca_axis", min_vol_pca )

	cylindricity_pca0 = segment.feature_dict["volume_by_res"] / segment.feature_dict["vol_cylinder_pca0"]
	cylindricity_pca1 = segment.feature_dict["volume_by_res"] / segment.feature_dict["vol_cylinder_pca1"]
	cylindricity_pca2 = segment.feature_dict["volume_by_res"] / segment.feature_dict["vol_cylinder_pca2"]

	segment.add_feature("cylindricity", max([cylindricity_pca0, cylindricity_pca1, cylindricity_pca2]))


	# entropy

	pca_eigenvalues = segment.feature_dict["pca_eigenvalues"]

	eig0 = pca_eigenvalues[0]/ sum(pca_eigenvalues)
	eig1 = pca_eigenvalues[1]/ sum(pca_eigenvalues)
	eig2 = pca_eigenvalues[2]/ sum(pca_eigenvalues)

	entropy = -  (eig0 * np.log2(eig0) + eig1 * np.log2(eig1) + eig2 * np.log2(eig2))

	segment.add_feature("entropy", entropy)
	


	

		

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
			
	
	

def hu_moments(mid_slice):
	
	if mid_slice.sum() <3:
		mid_slice = binary_dilation(mid_slice)

 	moments = cv2.moments(mid_slice.astype("uint8"), True)
	hu_moments = cv2. HuMoments(moments)
#	pdb.set_trace()
#	hu_moments = -np.sign(hu_moments)*np.log10(np.abs(hu_moments))


	return [x[0] for x in hu_moments]



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
	segment1.feature_dict["size_border_with_neighbor"].append((segment2.label, s))
	segment2.feature_dict["size_border_with_neighbor"].append((segment1.label, s))
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



def fit_line(segment):


	pts = np.nonzero(segment.mask)
	pts = np.array(pts).T

	pts_len = pts.shape[0]

	pts = pts.reshape((pts_len, 1, 3))
	
	line = cv2.fitLine(pts.astype("float32"), 1,0, 0.01, 0.01)

	line = [x[0] for x in line]

	return line

def tetrahedron_volume(a, b, c, d):

	return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def get_convex_hull_3d(segment):

#	t = time.time()
	pts = segment.border_coords_by_res
#	dt = Delaunay(pts)
#	tets = dt.points[dt.simplices]
#	vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))

#	print "v1", time.time() - t

#	t = time.time()
#	hull = ConvexHull(pts)
#	list_of_pts_indices = np.unique(hull.simplices)
#	new_pts = [pts[i] for i in list_of_pts_indices]
#	dt = Delaunay(new_pts)
#	tets = dt.points[dt.simplices]
#	vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))
#	print "v1", time.time() - t
#	
	#t = time.time()
	ch = ConvexHull(pts)
	vertex =  np.array([0])
	simplices = np.column_stack((np.repeat(vertex, ch.nsimplex), ch.simplices))
	tets = ch.points[simplices]
	vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]))
	#print "v2", time.time() -t

	#print vol, vol1

	segment.add_feature("convex_hull_volume", vol)
	segment.add_feature("vol_to_hull_vol_ratio", segment.feature_dict["volume_by_res"] / vol)
	

def get_convex_hull_properties_in_slice(segment):


	cnt = np.array(segment.get_mid_slice_contour())
	cnt_len = cnt.shape[0]
	cnt = cnt.reshape((cnt_len,1,2))

	if len(cnt)>3:

		hull = cv2.convexHull(cnt,returnPoints = False)
		

		segment.feature_dict["mid_slice_convex_hull_indices"] = [x[0] for x in hull]

		try:
			defects = cv2.convexityDefects(cnt,hull)
			if defects is not None:
				segment.feature_dict["mid_slice_convexity_deffects"] = [[i for i in defects[k][0]] for k in xrange(len(defects))]
			else: 
				segment.feature_dict["mid_slice_convexity_deffects"] = [[]]
		except:
			segment.feature_dict["mid_slice_convexity_deffects"] = [[]]


	else:
		segment.feature_dict["mid_slice_convex_hull_indices"] = []
		segment.feature_dict["mid_slice_convexity_deffects"] = [[]]
		

	

def get_nuclei_channel_features(segment, vol_nuclei):

	box_bounds = segment.bounding_box
	cropped_vol = vol_nuclei[box_bounds.xmin:box_bounds.xmax+1, box_bounds.ymin:box_bounds.ymax+1, box_bounds.zmin:box_bounds.zmax+1]

	hist = histogram_in_mask(cropped_vol, segment.mask) * segment.mask.sum()

	segment.add_feature("nuclei_channel_intensity_hist", hist)
	segment.add_feature("nuclei_channel_volume_with_highest_intensity", sum(hist[-2:]) * segment.xres * segment.yres * segment.zres)	



def init_neighbor_props_for_segment(segment):

	if not segment.feature_dict.has_key("percent_border_with_neighbor"):
		segment.feature_dict["percent_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("mean_intensity_border_with_neighbor"):
		segment.feature_dict["mean_intensity_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("size_border_with_neighbor"):
		segment.feature_dict["size_border_with_neighbor"] = []

	if not segment.feature_dict.has_key("weighted_merge_score"):
		segment.feature_dict["weighted_merge_score"] = []


def get_border_to_nucleus_properties(segment):


	pts = np.array(segment. isotropic_border_coords)
	res = min([segment.xres, segment.yres, segment.zres])
	centroid = np.floor(np.mean(pts,0))

	centroid_with_offset = np.floor(np.mean(segment.border_coords,0) + np.array([segment.bounding_box.xmin, segment.bounding_box.ymin, segment.bounding_box.zmin]))

	centroid_by_res = centroid_with_offset * np.array([segment.xres, segment.yres, segment.zres])

#	if centroid_with_offset[2] == 26:
#		pdb.set_trace()

	dists =  ((pts[:,0] - centroid[0])**2 + (pts[:,1] - centroid[1])**2 + (pts[:,2] - centroid[2])**2 ) **0.5
	max_dist = dists.max()
	dists = dists/max_dist	

	bins = np.arange(0, 1.05, 0.05)
	dist_hist = histogram(dists, bins = bins) [0]

	
	segment.add_feature("centroid_res", tuple(centroid_by_res))
	segment.add_feature("border_to_nucleus_dist_hist", dist_hist)
	segment.add_feature("border_to_nucleus_dist_mean", np.mean(dists))
	segment.add_feature("border_to_nucleus_dist_std", np.std(dists))
	segment.add_feature("distance_to_border_scale_factor", max_dist)


	pts = np.array(segment.border_coords)
	segment.add_feature("centroid", tuple(centroid_with_offset.astype("int")))

		


def get_segments_with_features(vol, label_map, set_of_labels, name_of_parent, nuclei_collection, vol_nuclei= None):

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

	x_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["x_res"])
	y_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["y_res"])
	z_res = float(CellECT.seg_tool.globals.DEFAULT_PARAMETER["z_res"])

	t1 = time.time()
	segment_collection = segc.SegmentCollection(set_of_labels, label_map, name_of_parent, x_res, y_res, z_res)
	t2 = time.time()
	print "....... %.3f sec                         " %(t2 - t1)
	logging.info ("... %.3f sec" % (t2-t1))

	

	message = "Getting properties for %d segments from %s ..." % (len(set_of_labels), name_of_parent)
	print message
	logging.info(message)

	dist_metric = None

#[[270, 200,15],[280, 400, 14], [280, 600,14], [290, 700,13]]

	# LEFT-RIGHT coordinate first
	# UP-DOWN coordinate second


	if name_of_parent == "test_volume":
		axis_pts = 	CellECT.seg_tool.globals.DEFAULT_PARAMETER["APaxis"] 
	
		list1, list2, list3 = zip (* axis_pts)
		ap_axis = APaxis.APaxis(list1, list2, list3, x_res, y_res, z_res, label_map.shape, label_map>1)


	if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_dist_from_margin"]) and \
       len(segment_collection.list_of_segments) and \
       segment_collection.list_of_segments[0].name_of_parent == "test_volume":
		dist_metric = DistanceFromMargin(label_map, x_res, y_res, z_res)
	
	t1 = time.time()

	add_nucleus_to_segments(segment_collection, nuclei_collection, label_map)

	segment_collection.make_contours_for_all_segments(label_map)

	sum_time = 0

	for segment in segment_collection.list_of_segments:
		if segment.mask.sum() < 5:
			continue
		try:
			if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_size"]):
				if should_compute_feature(segment.name_of_parent, "size"):
					segment.add_feature("size", segment.mask.sum())
					#segment.add_feature("size", len(segment.list_of_voxel_tuples))


#			t = time.time()
			if segment.name_of_parent == "test_volume":		


				init_neighbor_props_for_segment(segment)

				for neighbor_label in segment.neighbor_labels:
					seg_idx = segment_collection.segment_label_to_list_index_dict[neighbor_label]
					segment2 = segment_collection.list_of_segments[seg_idx]
					init_neighbor_props_for_segment(segment2)
					add_neighbor_border_properties(segment,segment2, vol )	

#			print "neighbors", time.time() -t

			box_bounds = segment.bounding_box
				
#			t = time.time()
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
#			print "border", time.time() -t

	
			if should_compute_feature(segment.name_of_parent, "inner_point"):
					segment.add_feature("inner_point", segment_inner_point(segment))

			if segment.name_of_parent == "test_volume":

				if vol_nuclei != None:
					get_nuclei_channel_features(segment, vol_nuclei)


	
#				t = time.time()
				segment.add_feature("mid_slice_hu_moments", hu_moments(segment.get_mid_slice()))
				segment.add_feature("mid_slice_best_contour", segment.get_mid_slice_contour())
				segment.add_feature("mid_slice_z", segment.mid_slice_z)


				segment.add_feature("line_fit", fit_line(segment))

				get_convex_hull_properties_in_slice(segment)
#				print "mid slice", time.time() - t
				
#				t = time.time()
				get_border_to_nucleus_properties(segment)
#				print "dist_hist:", time.time() - t
	
#				t = time.time()
				ap_axis.add_segment_projection_properties(segment)
#				print "ap", time.time() - t				

#				t = time.time()
				get_pca_properties(segment)

				get_min_oriented_bounding_box_properties(segment)
#				print "pca", time.time() - t

#				t = time.time()
				# TOO SLOW...... UNCOMMENT IF NEEDED				
				get_convex_hull_3d(segment)
#				print "hull", time.time() - t
			
		

				if int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["use_dist_from_margin"]):

					t = time.time()
					if should_compute_feature(segment.name_of_parent, "distance_from_margin"):
						#segment.add_feature("min_distance_from_margin", dist_metric.get_min_dist_for_segment(segment))
						#segment.add_feature("mean_distance_from_margin", dist_metric.get_mean_dist_for_segment(segment))
						#segment.add_feature("max_distance_from_margin", dist_metric.get_max_dist_for_segment(segment))
						segment.add_feature("centroid_dist_from_margin", dist_metric.get_centroid_dist(segment))
					sum_time += time.time() - t

#				print "........"
		except:
			pdb.set_trace()
		
		
		counter += 1
		misc.print_progress(counter, total)

		#print np.mean(segment.feature_dict["border_to_nucleus_distance"]), segment.nucleus.index
	t2 = time.time()
	print sum_time
	print "....... %.3f sec                           " % (t2 - t1)
	logging.info ("... %.3f sec" % (t2-t1))



	return segment_collection


