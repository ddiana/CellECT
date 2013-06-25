# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import pylab
import re
from termcolor import colored
import xml.etree.ElementTree as ET

# Imports from this project
import globals
from nuclei_collection import nuclei_collection as nc
from seed_collection import seed_collection as sc

"""
Functions to load saved data from xml files.
"""

def load_nuclei_from_xml():

	"Load nuclei collection from xml file."

	file_name = globals.DEFAULT_PARAMETER["save_location_prefix"] + "nuclei.xml"
	print colored("Loading nuclei from "+ file_name, "cyan")
	tree = ET.parse(file_name)

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
	set_size_string = union_find_field[0][0].text
	set_size_string = re.findall("(\d+)", set_size_string)
	set_size = [int(val) for val in set_size_string]

	nuclei_collection.union_find.parents = parents
	nuclei_collection.union_find.set_size = set_size


	return nuclei_collection


def load_seeds_from_xml():

	"Load seed collection from xml file."

	file_name = globals.DEFAULT_PARAMETER["save_location_prefix"] + "seeds.xml"

	print colored("Loading seeds from "+ file_name, "cyan")
	tree = ET.parse(file_name)

	root = tree.getroot()

	seed_collection = sc.SeedCollection([])

	for child in root:
		if child.tag == "seed":
			x = int(child.attrib["x"])
			y = int(child.attrib["y"])
			z = int(child.attrib["z"])
			index = int(child.attrib["index"])
			nucleus_index = int(child.attrib["nucleus_index"])

			nucleus = seed_collection.add_seed(sc.Seed(x,y,z,index,nucleus_index))


	return seed_collection


