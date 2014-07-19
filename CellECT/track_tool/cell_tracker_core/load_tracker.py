# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import xml.etree.ElementTree as ET
import pdb
import os.path

# Imports from this project
from CellECT.seg_tool.nuclei_collection.nuclei_collection import Nucleus
from CellECT.track_tool.cell_tracker_core import cell_profile as cp
from CellECT.track_tool.cell_tracker_core import cell_tracker as ct

import CellECT.track_tool.globals
import CellECT.seg_tool.globals
import CellECT.seg_tool.seg_utils.bounding_box as bbx

"""
Initialize cell tracker.
Extract cell profile information for every cell in every time stamp in the dataset.
"""


def parse_file_at_timestamp(file_name, timestamp):

	""" For a segmentation result from CellECT_seg_tool at a timestamp,
		read all the segment information and create a "cell profile" for each
		cell 

		file_name = file with the segment collection xml
		timestamp = timestamp corresponding to this segmentation
		
		Returns list of cell profiles.			
	"""

	# parse xml file with segment collection
	tree = ET.parse(file_name)
	xml_list_of_segments = tree.getroot().find("list_of_segments")
	
	list_of_cell_profiles = []	

	# make cell profile for each cell

	nan = None

	for segment in xml_list_of_segments.findall("segment"):
		
		xml_nucleus = segment.find("nucleus")

		x = float( xml_nucleus.attrib["x"] )
		y = float( xml_nucleus.attrib["y"] )
		z = float( xml_nucleus.attrib["z"] )
		index = float( xml_nucleus.attrib["index"] )

		seg_nucleus = Nucleus( x,y,z,index )
		seg_label = int(segment.attrib["label"])

		

		xml_bbx = segment.find("bounding_box")

		xmin = float( xml_bbx.attrib["xmin"] )
		ymin = float( xml_bbx.attrib["ymin"] )
		zmin = float( xml_bbx.attrib["zmin"] )
		xmax = float( xml_bbx.attrib["xmax"] )
		ymax = float( xml_bbx.attrib["ymax"] )
		zmax = float( xml_bbx.attrib["zmax"] )


		seg_bbx = bbx.BoundingBox(xmin,xmax, ymin, ymax, zmin, zmax)
	


		#features_of_interest = CellECT.seg_tool.globals.specific_features
		segment_feature = {}
		feat_dict = segment.find("feature_dictionary").findall("feature")

		# if inner point given replace nucleus with this

		inner_point_xml = res = filter(lambda x: x.attrib["name"] == "inner_point", feat_dict)
		if len(inner_point_xml):		
			inner_point_tuple = eval (inner_point_xml[0].text)
			set_nucleus = Nucleus (inner_point_tuple[0], inner_point_tuple[1], inner_point_tuple[2], index)
		

		res = filter(lambda x: x.attrib["name"] == "weighted_merge_score", feat_dict)

		neighbor_labels = [x[0] for x in eval(res[0].text)]


		feat = "mid_slice_hu_moments"
		res = filter(lambda x: x.attrib["name"] == feat, feat_dict)
		try:
			segment_feature[feat] = eval(res[0].text)
		except: 
			segment_feature[feat] = None

		feat = "mid_slice_best_contour"
		res = filter(lambda x: x.attrib["name"] == feat, feat_dict)
		segment_feature[feat] = eval(res[0].text)

		feat = "border_to_nucleus_dist_hist"
		res = filter(lambda x: x.attrib["name"] == feat, feat_dict)
		segment_feature[feat] = [int(x) for x in res[0].text.strip('[').strip(']').split()]



		features_of_interest = ["surface_area_by_res", "sphericity","volume_by_res", "entropy", "cylindricity", "flatness", "elongation","volume_by_res_to_enclosing_sphere_vol_ratio", "surface_area_by_res", "dist_to_AP_axis", "angle_with_AP_axis", "position_along_AP_axis", "size", "centroid_dist_from_margin", "vol_to_hull_vol_ratio", "centroid_res"]

		for feat in features_of_interest:
			res = filter(lambda x: x.attrib["name"] == feat, feat_dict)

			value = eval(res[0].text)

			segment_feature[feat] = value


#		for feat in features_of_interest:


#			res = filter(lambda x: x.attrib["name"] == feat, feat_dict)
#			if res:

#				try:
#					# is this a float..
#					segment_feature[feat] = float(res[0].text)
#				except:
#					# is this a list...
#	
#					temp = []
#					for item in res[0].text.strip('[').strip(']').split():
#						try:	
#							temp.append( float( item)) 
#						except:
#							pass

#					if len(temp):
#						segment_feature[feat] = temp

		res = filter(lambda x: x.attrib["name"] == "size", feat_dict)
		seg_size = int(res[0].text)

		cell_profile = cp.CellProfile(seg_label, seg_nucleus, seg_size, seg_bbx, neighbor_labels ,segment_feature)
		
		list_of_cell_profiles.append(cell_profile)

	return list_of_cell_profiles



def load_cell_tracker():

	""" 
	Initialize empty cell tracker.
	For every time stamp in the dataset, extract the cell profile for every
	cell in every segmentation.
	Build graph based of cell tracker.
	"""


	cell_tracker = ct.CellTracker()
	time_stamps = CellECT.track_tool.globals.PARAMETER_DICT["time-stamps"]

	for t in time_stamps:
		cell_profiles = parse_file_at_timestamp( os.path.join(CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] , "timestamp_" + str(t) +"_segment_props.xml"),t)
		cell_profile_per_ts = cp.CellProfilesPerTimestamp(t,cell_profiles)
		cell_tracker.add_cell_profiles_per_timestamp(cell_profile_per_ts)


	if CellECT.track_tool.globals.PARAMETER_DICT["with_tracker"]:
		cell_tracker.build_lineage()
	

	return cell_tracker


