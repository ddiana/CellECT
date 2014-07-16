# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import scipy as sp
from scipy import io
import pdb
import logging
from termcolor import colored

# Imports from this project
from CellECT.seg_tool.seg_io import load_xml
import CellECT.seg_tool.globals

"""
Functions to load last save.
"""

def load_last_save():

	"Load .mat files and .xml files."

	logging.info("Loading last save.")	
	
	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "label_map.mat"

	try:
		nuclei_collection = load_xml.load_nuclei_from_xml()
		seed_collection = load_xml.load_seeds_from_xml()
		watershed = io.loadmat(file_name)["ws"]
		vol = io.loadmat(file_name)["vol"]
		bg_seeds = load_xml.load_bg_seeds_xml()
	except Exception as err:
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		logging.exception(err)
		logging.exception(err.message)
		sys.exit()

	bg_prior = None
	try:
		bg_prior =  io.loadmat(file_name)["bg_mask"]
	except:
		print "No background prior."

	return nuclei_collection, seed_collection, watershed, bg_seeds, bg_prior, vol




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


