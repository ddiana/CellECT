import os
import sys


def prepare_directories(workspace_path):

	if os.path.exists(workspace_path):
		print workspace_path, "already exists... Did not create directories."
	else:
		os.makedirs(workspace_path + "/config_files")
		os.makedirs(workspace_path + "/init_watershed_all_time_stamps")
		os.makedirs(workspace_path + "/input_slices")
		os.makedirs(workspace_path + "/segs_all_time_stamps")
		os.makedirs(workspace_path + "/tracker_config")
		os.makedirs(workspace_path + "/training_data")


if __name__ == "__main__":

	if len(sys.argv) != 2:
		print "Usage: python create_workspace_directories.py WORSPACE_PATH"
		print "Example: python create_workspace_directories.py ascidian_workspace"
		print "	   This will create the ascidian_workspace directory in the current directory and place the necessary directories in it."
	else:
		prepare_directories(sys.argv[1])
