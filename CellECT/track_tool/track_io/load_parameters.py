# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import re
import pdb
import os
import os.path

# Imports from this project
import CellECT.track_tool.globals


"""
Load parameters from the config file.
Prepare global parameters.
Load defaults, if applicable.
"""


def get_time_stamps_from_input_string(input_string):

	"""
	Time stamps come as a list of values in the config file, separated by ' '
	"""

	CellECT.track_tool.globals.PARAMETER_DICT["time-stamps"] = [int(number.strip()) for number in input_string.split(" ")]



def abs_path_to_workspace(config_file_path):

	"""
	Given the relative path to the config file, return the absolute path to the
	workspace directory
	"""

	abs_path_to_config_file = os.path.abspath(config_file_path)

	# config file is always in the config_files directory of the workspace directory
	abs_path_to_workspace_dir = os.path.join(os.path.abspath(os.path.join(abs_path_to_config_file , "..", "..")), "")
	
	if not os.path.isdir(abs_path_to_workspace_dir):
		err = IOError("Could not find workspace directory.")
		raise err



	print "Workspace at: %s" % abs_path_to_workspace_dir

	return abs_path_to_workspace_dir


def make_abs_path(path_to_ws, relative_path):

	return os.path.join (path_to_ws , relative_path)

def adjust_abs_path():

	
	CellECT.track_tool.globals.PARAMETER_DICT["segs-path"] = make_abs_path(CellECT.track_tool.globals.abs_path_to_workspace_dir, CellECT.track_tool.globals.PARAMETER_DICT["segs-path"])
	CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"] = make_abs_path(CellECT.track_tool.globals.abs_path_to_workspace_dir, CellECT.track_tool.globals.PARAMETER_DICT["tif-slices-path"])


def read_program_parameters(config_file_path):

	# TODO unit tests for this

	try:
		f = open(config_file_path)
	except IOError as err:
		err .message = "Could not read config file at:" + config_file_path 
		logging.exception(err)
		logging.exception(err.message)
		print colored("Error: %s " % err.message , 'red')
		sys.exit()

	line = f.readline()
	
	while (line != ""):
		matches = re.match("(.*)=(.*)", line)
		if matches:
			key = matches.group(1).strip()
			val = matches.group(2).strip()
			CellECT.track_tool.globals.PARAMETER_DICT[key] = val
		else:
			print "ERROR reading config file"
		line = f.readline()

	CellECT.track_tool.globals.abs_path_to_workspace_dir = abs_path_to_workspace(config_file_path)
	

	adjust_abs_path()
	get_time_stamps_from_input_string(CellECT.track_tool.globals.PARAMETER_DICT["time-stamps"])

