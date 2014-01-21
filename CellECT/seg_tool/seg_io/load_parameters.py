# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import pdb
import re
import sys
from termcolor import colored
import os
import logging

# Imports from this project
import CellECT.seg_tool.globals 




def parse_config_file_line(line, required_keys, all_keys):
	"""
	Parse a line from the config fine: key = value.
	"""

	if not line.strip():
		return None, None
	
	matches = re.match("(.+)=(.+)", line)



	if matches:
		key = matches.group(1).strip().lower()
		val = matches.group(2).strip()

		if not key in all_keys:
			
			err = IOError()
			err.message = "Key '%s' not in parameter list" % key
			raise err

		if not val:
			err = IOError()
			err.message = "No value assigned for key '%s'" % key
			raise err
	else:
		err = IOError()
		err.message = "No key=value pattern."
		logging.exception(err.message)
		raise err
		return None, None


	if CellECT.seg_tool.globals.DEFAULT_PARAMETER.has_key(key):
		err = IOError()
		err.message = "Redefinition of key '%s'" % key
		raise err
			
	return key, val


def check_if_file_complete(required_keys):
	"""
	Check if the config file addressed all the parameters needed.
	"""

	missing_info = ""

	for key in required_keys:
		if not CellECT.seg_tool.globals.DEFAULT_PARAMETER.has_key(key):
			missing_info += key + " "


	if len(missing_info):
		err = IOError()
		err.message = "Error reading config file: incomplete config file. Minssing info: "+missing_info
		raise err





def read_program_parameters(config_file_path):

	"Read parameters from config file."

	try:
		f = open(config_file_path)
	except IOError as err:
		err .message = "Could not read config file at:" + config_file_path 
		logging.exception(err)
		logging.exception(err.message)
		print colored("Error: %s " % err.message , 'red')
		sys.exit()


	required_keys = set(CellECT.seg_tool.globals.default_parameter_dictionary_keys)

	if CellECT.seg_tool.globals.DEFAULT_PARAMETER["bisque"]:
		required_keys = required_keys.union( set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_bq_only))
	if not CellECT.seg_tool.globals.DEFAULT_PARAMETER["no_cellness_metric"]:
		required_keys = required_keys.union( set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_cellness_metric_only))

	all_keys = set(CellECT.seg_tool.globals.default_parameter_dictionary_keys)
	all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_bq_only))
	all_keys = all_keys.union(set(CellECT.seg_tool.globals.default_parameter_dictionary_keys_cellness_metric_only))



	line = f.readline()
	line_counter = 0

	while (line != ""):
		line_counter += 1

		try:
			key, val = parse_config_file_line(line, required_keys, all_keys)
			
		except IOError as err:
			message = "Error at line #%d: %s \nLine #%d: %s" % (line_counter, err.message, line_counter, line.strip() )
			print colored(message, "red")
			logging.exception(message)
			sys.exit()

		except Exception as err:
			message = "Error at line #%d: %s \nLine #%d: %s" % (line_counter, err.message, line_counter, line.strip() )
			logging.exception(message)
			print colored(message, "red")
			sys.exit()
			
		if key and val:
			CellECT.seg_tool.globals.DEFAULT_PARAMETER[key] = val
		line = f.readline()

	try:
		check_if_file_complete(required_keys)
	except IOError as err:
		message = "Error in the config file: %s" % err.message
		logging.exception(message)
		print colored(message, "red")		
		sys.exit()


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

	print "Workspace at: %s" % abs_path_to_workspace_dir

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
		message = "Error: %s" % err.message
		print colored(message)
		logging.exception(err)
		logging.exception(err.message)
		sys.exit()

	# convert everything to absolute path
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_nuclei_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["volume_nuclei_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["first_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["first_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["nuclei_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["nuclei_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_vol_nuclei_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_positive_labels_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_seg_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["training_negative_labels_mat_path"])
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] = make_absolute_path(CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"])
	
	# convert to number
	CellECT.seg_tool.globals.DEFAULT_PARAMETER["bisque"] = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["bisque"])


	logging.info("DEFAULT PARAMETERS: %s" % CellECT.seg_tool.globals.DEFAULT_PARAMETER)

	
