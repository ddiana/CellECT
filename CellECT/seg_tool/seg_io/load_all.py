# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import scipy as sp
from scipy import io
import pdb

# Imports from this project
from CellECT.seg_tool.seg_io import load_xml
import CellECT.seg_tool.globals

"""
Functions to load last save.
"""

def load_last_save():
	
	"Load .mat files and .xml files."	
	
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "label_map.mat"

	try:
		nuclei_collection = load_xml.load_nuclei_from_xml()
		seed_collection = load_xml.load_seeds_from_xml()
		watershed = io.loadmat(file_name)["ws"]
	except Exception as err:
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		sys.exit()

	return nuclei_collection, seed_collection, watershed




def load_from_mat(file_name, var_name):

	"Load from mat files."
	
	try:
		all_variables = sp.io.loadmat(file_name)
	except IOError as err:
		err.message = "Could not open file at %s" % file_name
		raise err


	try:
		var = all_variables[var_name]
	except KeyError as err:
		err.message = "Could not find variable %s in file %s" % (var_name, file_name)
		raise err



	return  var


