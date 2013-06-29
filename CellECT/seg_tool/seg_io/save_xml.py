# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import xml.etree.ElementTree as ET

# Imports from this project
import CellECT.seg_tool.globals

"""
Functions to save xml info: nuclei, segments, seeds, seed_segments.
"""

def save_xml_file_nuclei(nuclei_collection):

	"Save nuclei collection in xml file."
	
	root = ET.Element("nuclei_collection")
	
	for nucleus in nuclei_collection.nuclei_list:
		attributes = {"x": str(nucleus.x), "y": str(nucleus.y), "z": str(nucleus.z), "index": str(nucleus.index), "added_by_user": str(nucleus.added_by_user)}
		nucleusElem = ET.SubElement(root,"nucleus", attrib=attributes)

	union_find = ET.SubElement(root,"union_find")	
	union_find_parents = ET.SubElement(union_find, "parents")
	union_find_parents.text = str(nuclei_collection.union_find.parents)
	union_find_set_size = ET.SubElement(union_find, "set_size")
	union_find_set_size.text = str(nuclei_collection.union_find.set_size)
	
	tree = ET.ElementTree(root)
	
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "nuclei.xml"

	try:
		tree.write(file_name)
	except IOError as err:
		err.message = "Error saving xml tree at %s" % file_name
		raise err

	print "....... Nuclei XML file at:", file_name
	

def save_xml_file_seeds(seed_collection):

	"Save seed collection in xml file."
	
	root = ET.Element("seed_collection")
	
	for seed in seed_collection.list_of_seeds:
		attributes = {"x": str(seed.x), "y": str(seed.y), "z": str(seed.z), "index": str(seed.index), "nucleus_index": str(seed.nucleus_index)}
		nucleusElem = ET.SubElement(root,"seed", attrib=attributes)

	tree = ET.ElementTree(root)
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "seeds.xml"

	try:
		tree.write(file_name)
	except IOError as err:
		err.message = "Error saving xml tree at %s" % file_name
		raise err

	print "....... Seeds XML file at:", file_name

	
def save_xml_file_segment_props(segment_collection):

	"Save segment collection in xml file."

	root = ET.Element("segment_collection")
	
	info = ET.SubElement(root,"info")
	parent_field = ET.SubElement(info,"name_of_parent")
	parent_field.text = segment_collection.name_of_parent

	list_of_segments_field = ET.SubElement(root, "list_of_segments")
	
	for segment in segment_collection.list_of_segments:
		segment_field = ET.SubElement(list_of_segments_field, "segment")
		segment_field.set("label", str(segment.label))

		
		nucleus_field = ET.SubElement(segment_field, "nucleus")
		nucleus_field.set("x",str( segment.nucleus_list[0].x))
		nucleus_field.set("y",str( segment.nucleus_list[0].y))
		nucleus_field.set("z",str( segment.nucleus_list[0].z))
		nucleus_field.set("index",str( segment.nucleus_list[0].index))
		nucleus_field.set("added_by_user",str( segment.nucleus_list[0].added_by_user))

		bounding_box_field = ET.SubElement(segment_field, "bounding_box")
		bounding_box_field.set("xmin", str(segment.bounding_box.xmin))
		bounding_box_field.set("xmax", str(segment.bounding_box.xmax))
		bounding_box_field.set("ymin", str(segment.bounding_box.ymin))
		bounding_box_field.set("ymax", str(segment.bounding_box.ymax))
		bounding_box_field.set("zmin", str(segment.bounding_box.zmin))
		bounding_box_field.set("zmax", str(segment.bounding_box.zmax))
	

		feature_dict = ET.SubElement(segment_field, "feature_dictionary")
		
		for key in segment.feature_dict.keys():
			feature_field = ET.SubElement(feature_dict, "feature")
			feature_field.set("name", key )
			feature_field.text = str(segment.feature_dict[key])


	tree = ET.ElementTree(root)
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "segment_props.xml"

	try:
		tree.write(file_name)
	except IOError as err:
		err.message = "Error saving xml tree at %s" % file_name
		raise err

	print "....... Segment properties XML file at:", file_name


def save_xml_file_seed_segment_props(seed_segment_collection):

	"Save seed-segment collection in xml file."

	root = ET.Element("segment_collection")
	
	info = ET.SubElement(root,"info")

	list_of_seed_segments_field = ET.SubElement(root, "list_of_seed_segments")
	
	for seed_segment in seed_segment_collection.list_of_seed_segments:
		seed_segment_field = ET.SubElement(list_of_seed_segments_field, "seed_segment")

		
		seed_field = ET.SubElement(seed_segment_field, "seed")
		seed_field.set("x",str( seed_segment.seed.x))
		seed_field.set("y",str( seed_segment.seed.y))
		seed_field.set("z",str( seed_segment.seed.z))
		seed_field.set("index",str( seed_segment.seed.index))
		seed_field.set("nucleus_index",str( seed_segment.seed.nucleus_index))
	
		bounding_box_field = ET.SubElement(seed_segment_field, "bounding_box")
		bounding_box_field.set("xmin", str(seed_segment.bounding_box.xmin))
		bounding_box_field.set("xmax", str(seed_segment.bounding_box.xmax))
		bounding_box_field.set("ymin", str(seed_segment.bounding_box.ymin))
		bounding_box_field.set("ymax", str(seed_segment.bounding_box.ymax))
		bounding_box_field.set("zmin", str(seed_segment.bounding_box.zmin))
		bounding_box_field.set("zmax", str(seed_segment.bounding_box.zmax))

		feature_dict = ET.SubElement(seed_segment_field, "feature_dictionary")
		


	tree = ET.ElementTree(root)
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "seed_segment_props.xml"

	try:
		tree.write(file_name)
	except IOError as err:
		err.message = "Error saving xml tree at %s" % file_name
		raise err

	print "....... Seed segment properties XML file at:", file_name



