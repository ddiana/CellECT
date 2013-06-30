# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import re
import sys
from termcolor import colored
import os

# Imports from this project
import CellECT.seg_tool.globals 


def parse_config_file_line(line):
	"""
	Parse a line from the config fine: key = value.
	"""

	if not line.strip():
		return None, None
	
	matches = re.match("(.+)=(.+)", line)


	if matches:
		key = matches.group(1).strip().lower()
		val = matches.group(2).strip()

		if not key in CellECT.seg_tool.globals.default_parameter_dictionary_keys:
			raise IOError("Key '%s' not in parameter list" % key)

		if not val:
			raise IOError("No value assigned for key '%s'" % key)
	else:
		raise IOError("No key=value pattern.")
		return None, None


	if CellECT.seg_tool.globals.DEFAULT_PARAMETER.has_key(key):
		raise IOError("Redefinition of key '%s'" % key)
			
	return key, val


def check_if_file_complete():
	"""
	Check if the config file addressed all the parameters needed.
	"""

	missing_info = ""
	for key in CellECT.seg_tool.globals.default_parameter_dictionary_keys:
		if not CellECT.seg_tool.globals.DEFAULT_PARAMETER.has_key(key):
			missing_info += key + " "

	if len(missing_info):
		raise IOError("Error reading config file: incomplete config file. Minssing info: "+missing_info)



def read_program_parameters(config_file_path):

	"Read parameters from config file."

	try:
		f = open(config_file_path)
	except IOError as err:
		print colored("Could not read config file at:" + config_file_path, 'red')
		sys.exit()


	line = f.readline()
	line_counter = 0

	while (line != ""):
		line_counter += 1

		try:
			key, val = parse_config_file_line(line)
			
		except IOError as err:
			print colored("Error at line #%d: %s \nLine #%d: %s" % (line_counter, err.message, line_counter, line.strip() ), "red")
			sys.exit()
			# TODO: write to log file
		except Exception as err:
			print colored("Error at line #%d: %s \nLine #%d: %s" % (line_counter, err.message, line_counter, line.strip() ), "red")
			sys.exit()
			# TODO: write to log file
			
		if key and val:
			CellECT.seg_tool.globals.DEFAULT_PARAMETER[key] = val
		line = f.readline()

	try:
		check_if_file_complete()
	except IOError as err:
		print colored("Error in the config file: %s" % err.message, "red")		
		sys.exit()
		# TODO: write to log file

	prepare_program_parameters(config_file_path)



def abs_path_to_workspace(config_file_path):

	"""
	Given the relative path to the config file, return the absolute path to the
	workspace directory
	"""

	abs_path_to_config_file = os.path.abspath(config_file_path)

	# config file is always in the config_files directory of the workspace directory
	abs_path_to_workspace_dir = os.path.abspath(abs_path_to_config_file + "/../..") + "/"
	
	if not os.path.isdir(abs_path_to_workspace_dir):
		err = IOError("Could not find workspace directory.")
		raise err

	test_workspace_directory_structure(abs_path_to_workspace_dir)		

	print abs_path_to_workspace_dir

	return abs_path_to_workspace_dir

def test_workspace_directory_structure(path_to_workspace):
	"""
	Given the path to the workspace directory, check that the necessary 
	directories are present.
	"""

	for directory in CellECT.seg_tool.globals.expected_workspace_directories:
		assert os.path.isdir("%s/%s" % (path_to_workspace, directory)), colored("Bad woskspace structure. Directory %s not found in workspace at %s." % (directory, path_to_workspace), 'red')



def make_absolute_path(path_from_config_file):

	"""
	convert path relative to workspace to absolute path
	"""

	return CellECT.seg_tool.globals.path_to_workspace + path_from_config_file


def prepare_program_parameters(config_file_path):

	"""
	Given the path the user gave for the config file, which is in 
	...../workspace/config_files/ etc find the path to workspace only.
	"""

	# get absolute path to workspace
	try:
		CellECT.seg_tool.globals.path_to_workspace = abs_path_to_workspace(config_file_path)
	except Exception as err:
		print colored("Error: %s" % err)
		sys.exit()

	# convert everything to absolute path
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["first_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["first_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["nuclei_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["nuclei_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"])
	

	
