# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import re
from termcolor import colored
import xml.etree.ElementTree as ET
import sys

# Imports from this project
import CellECT.seg_tool.globals
from CellECT.seg_tool.nuclei_collection import nuclei_collection as nc
from CellECT.seg_tool.seed_collection import seed_collection as sc

"""
Functions to load saved data from xml files.
"""

def load_bg_seeds_xml():

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "bg_seeds.xml"
	
	print colored("Loading background seeds from "+ file_name, "cyan")
	
	try:
		tree = ET.parse(file_name)
	except IOError as err:
		err.message = "Could not open nuclei xml file at %s" % file_name
		raise err

	root = tree.getroot()
	bg_seeds = eval(tree.findall("list_of_seeds")[0].text)

	return bg_seeds

def load_nuclei_from_xml():

	"Load nuclei collection from xml file."

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "nuclei.xml"
	
	print colored("Loading nuclei from "+ file_name, "cyan")
	
	try:
		tree = ET.parse(file_name)
	except IOError as err:
		err.message = "Could not open nuclei xml file at %s" % file_name
		raise err


	root = tree.getroot()

	nuclei_collection = nc.NucleusCollection([])

	for child in root:
		if child.tag == "nucleus":
			x = int(child.attrib["x"])
			y = int(child.attrib["y"])
			z = int(child.attrib["z"])
			index = int(child.attrib["index"])
			added_by_user = False
			if child.attrib["added_by_user"] == "True":
				added_by_user = True
			nucleus = nuclei_collection.add_nucleus(nc.Nucleus(x,y,z,index,added_by_user))

	union_find_field = tree.findall("union_find")
	parents_string = union_find_field[0][0].text
	parents_string = re.findall("(\d+)", parents_string)
	parents = [int(val) for val in parents_string]

	set_size_string = union_find_field[0][1].text
	set_size_string = re.findall("(\d+)", set_size_string)
	set_size = [int(val) for val in set_size_string]

	try:
		is_deleted_string = union_find_field[0][2].text
		is_deleted_string = re.findall("(\d+)", is_deleted_string)
		is_deleted = [int(val) for val in is_deleted_string]
	except:
		is_deleted = [0 for val in set_size]

	nuclei_collection.union_find.parents = parents
	nuclei_collection.union_find.set_size = set_size
	nuclei_collection.union_find.is_deleted = is_deleted


	return nuclei_collection


def load_seeds_from_xml():

	"Load seed collection from xml file."

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "seeds.xml"


	
	print colored("Loading seeds from "+ file_name, "cyan")
	try:
		tree = ET.parse(file_name)
	except IOError as err:
		err. message = "Could not open seeds xml file at %s" % file_name
		raise err




	root = tree.getroot()

	seed_collection = sc.SeedCollection([])

	for child in root:
		if child.tag == "seed":
			x = int(child.attrib["x"])
			y = int(child.attrib["y"])
			z = int(child.attrib["z"])
			index = int(child.attrib["index"])
			nucleus_index = int(child.attrib["nucleus_index"])

			nucleus = seed_collection.add_seed(sc.Seed(x,y,z,nucleus_index, index))


	return seed_collection


