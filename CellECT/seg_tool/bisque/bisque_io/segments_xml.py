# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import xml.etree.ElementTree as ET
import random
import copy

# Imports from this project
import CellECT.seg_tool.globals


def write_bisque_segments_xml(nuclei_collection, seed_collection, segment_collection, seed_segment_collection, label_map):

	"Save nuclei collection in xml file."


	MAX_SEGMENTS = 100

	segment_collection.make_contours_for_all_segments(label_map)
	time_stamp = CellECT.seg_tool.globals.DEFAULT_PARAMETER["bq_t"]

	root = ET.Element("gobject", attrib = {"type":"CellECT_segments"})

	counter = 0
	shuffled_list_of_segments = copy.deepcopy(segment_collection.list_of_segments)
	random.shuffle(shuffled_list_of_segments)
	for segment in shuffled_list_of_segments:

		if counter > MAX_SEGMENTS:
			break


		attributes = {"type": "segment", "name": str(segment.label)}
		segment_field = ET.SubElement(root,"gobject", attrib=attributes)

		# print segment contours

		poly_counter = 0
		for polygon in segment.contour_polygons_list:
			
			polygon_field = ET.SubElement(segment_field, "polygon", attrib={"name": ("segment%d_poly%d" % (counter, poly_counter))})
			poly_counter += 1
			v_index = 0
			for vertex in polygon:
				attributes = {"index": str(v_index), "x": str(vertex[1]), "y": str(vertex[0]), "z": str(vertex[2]), "t": str(time_stamp)}	
				vertex_field = ET.SubElement(polygon_field, "vertex", attrib= attributes)
				v_index += 1

		feature_dict = ET.SubElement(segment_field, "tag", attrib={"name":"feature"})
		
		# print the features associated with this segment
		for key in segment.feature_dict.keys():
			feature_field = ET.SubElement(feature_dict, "tag", attrib={"name":key, "value":str(segment.feature_dict[key])})
#			feature_field.set("name", key )
#			feature_field.set("value", segment.feature_dict[key] )
#			value_field = ET.SubElement(feature_field, "value",)
#			value_field.text = str(segment.feature_dict[key])
		
		# get the nucleus associated with this segment
		for nucleus in segment.nucleus_list:
			attributes = {"name": "seed","type": "nucleus"}
			seed_field = ET.SubElement(segment_field,"gobject", attrib=attributes)
			ET.SubElement(seed_field,"tag", attrib={"name":"added_by_user", "value":str(nucleus.added_by_user) })
			ET.SubElement(seed_field,"tag", attrib={"name":"index", "value":str(nucleus.index)})
			ET.SubElement(seed_field,"vertex", attrib={"x": str(nucleus.x), "y": str(nucleus.y), "z": str(nucleus.z), "t": str(time_stamp)})

			

		nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[nucleus.index]
		set_of_current_nucleus = nuclei_collection.union_find.find ( nucleus_list_pos )
		size_of_set_of_current_nucleus = nuclei_collection.union_find.set_size[nucleus_list_pos]

		# TODO: other nuclei from mergers, do this more efficiently		
		# if there are other nuclei merged with this segment, then get these nuclei
		if size_of_set_of_current_nucleus> 0:
			# find the other nuclei 
			for nucleus_list_pos in xrange (len(nuclei_collection.nuclei_list)):
				if nuclei_collection.union_find.find(nucleus_list_pos) == set_of_current_nucleus:
					ncl = nuclei_collection.nuclei_list[nucleus_list_pos]
					attributes = {"name": "seed", "type": "nucleus"}
					seed_field = ET.SubElement(segment_field,"gobject", attrib=attributes)
					ET.SubElement(seed_field,"tag", attrib={"name":"added_by_user", "value":str(nucleus.added_by_user) })
					ET.SubElement(seed_field,"tag", attrib={"name":"index", "value":str(nucleus.index)})
					ET.SubElement(seed_field,"vertex", attrib={"x": str(nucleus.x), "y": str(nucleus.y), "z": str(nucleus.z), "t": str(time_stamp)})
				# TODO: make a quick reverse index for this instead of O(n) search:

		# get the seeds associated with this segment, if any 
		# iterate through seeds, and check the head nucleus of their parent nucleus
		for seed in seed_collection.list_of_seeds:
			nucleus_index_for_seed = seed.nucleus_index 
			nucleus_for_seed_list_pos = nuclei_collection.nucleus_index_to_list_pos[nucleus_index_for_seed]
			set_of_seed_nucleus = nuclei_collection.union_find.find ( nucleus_for_seed_list_pos )
			if set_of_current_nucleus == set_of_seed_nucleus:
				attributes = {"name": "seed","type": "seed", "x": str(seed.x)}
				seed_field = ET.SubElement(segment_field,"gobject", attrib=attributes)
				ET.SubElement(seed_field,"tag", attrib={"name":"nucleus_index", "value":str(seed.nucleus_index)})
				ET.SubElement(seed_field,"tag", attrib={"name":"index", "value": str(seed.index)})
				ET.SubElement(seed_field,"vertex", attrib={"x": str(seed.x), "y": str(seed.y), "z": str(nucleus.z), "t": str(time_stamp)})



		counter+=1

		

	tree = ET.ElementTree(root)
	
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "bisque.xml"

	try:
		tree.write(file_name)
	except IOError as err:
		err.message = "Error saving xml tree at %s" % file_name
		raise err

	print "....... Bisque XML file at:", file_name
