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

	nuclei_collection = load_xml.load_nuclei_from_xml()
	seed_collection = load_xml.load_seeds_from_xml()

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "label_map.mat"
	watershed = io.loadmat(file_name)["ws"]

	return nuclei_collection, seed_collection, watershed




def load_from_mat(file_name, var_name):

	"Load from mat files."

	return sp.io.loadmat(file_name)[var_name]


