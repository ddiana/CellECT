# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import xml.etree.ElementTree as ET
import pdb

# Imports from this project
from CellECT.seg_tool.nuclei_collection.nuclei_collection import Nucleus
from CellECT.track_tool.cell_tracker_core import cell_profile as cp
from CellECT.track_tool.cell_tracker_core import cell_tracker as ct

import CellECT.track_tool.globals
import CellECT.seg_tool.globals


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

	for segment in xml_list_of_segments.findall("segment"):
		
		xml_nucleus = segment.find("nucleus")

		x = float( xml_nucleus.attrib["x"] )
		y = float( xml_nucleus.attrib["y"] )
		z = float( xml_nucleus.attrib["z"] )
		index = float( xml_nucleus.attrib["index"] )

		seg_nucleus = Nucleus( x,y,z,index )
		seg_label = int(segment.attrib["label"])


		features_of_interest = CellECT.seg_tool.globals.specific_features
		segment_feature = {}
		feat_dict = segment.find("feature_dictionary").findall("feature")
		for feat in features_of_interest:
			res = filter(lambda x: x.attrib["name"] == feat, feat_dict)
			if res:

				try:
					# is this a float..
					segment_feature[feat] = float(res[0].text)
				except:
					# is this a list...
	
					temp = []
					for item in res[0].text.strip('[').strip(']').split():
						try:	
							temp.append( float( item)) 
						except:
							pass

					if len(temp):
						segment_feature[feat] = temp

		res = filter(lambda x: x.attrib["name"] == "size", feat_dict)
		seg_size = int(res[0].text)

		cell_profile = cp.CellProfile(seg_label, seg_nucleus, seg_size, segment_feature)
		
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
		cell_profiles = parse_file_at_timestamp( CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] +"/"+ "timestamp_" + str(t) +"_segment_props.xml",t)
		cell_profile_per_ts = cp.CellProfilesPerTimestamp(t,cell_profiles)
		cell_tracker.add_cell_profiles_per_timestamp(cell_profile_per_ts)

	pdb.set_trace()

	cell_tracker.build_lineage()
	

	return cell_tracker


