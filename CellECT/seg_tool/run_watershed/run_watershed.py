# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import os
from scipy import io as spio
import time
import numpy as np
import pdb
import sys
from termcolor import colored
import subprocess
import tempfile

# Imports from this project
import CellECT.seg_tool.globals
from CellECT.seg_tool.seg_utils import call_silent


"""
This module prepares input and calls watershed from Matlab.
TODO: Rplace this with custom C-extension of watershed algorithm to remove the
need for matlab.
"""

def run_watershed(vol, init_pts):

	"""
	This function calls matlab to run watershed on the given volume.
	TODO: Replace this with C extension of watershed.
	"""

	path_to_temp = tempfile.mkdtemp()


	
	print "Running seeded watershed...."
	has_bg = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["has_bg"])

	save_mat_file = "%s/watershed_input.mat" % path_to_temp
	try:
		call_silent.call_silent_err(spio.savemat, save_mat_file, {"vol":vol, "seeds": init_pts, "has_bg": has_bg})
	except Exception as err:
		err.message = "Could not write input file for Matlab at %s" % save_mat_file
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		sys.exit()


	t = time.time()
	
	matlab_file_path = CellECT.__path__[0] + "/utils"

	with open(os.devnull, "wb") as devnull:
		subprocess.check_call( ["matlab", "-nodesktop", "-nosplash", "-r", "cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')" % (matlab_file_path, path_to_temp, path_to_temp)], stdout=devnull, stderr=subprocess.STDOUT)

#	subprocess.check_call( ["matlab", "-nodesktop", "-nosplash", "-r", "cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')" % (matlab_file_path, path_to_temp, path_to_temp)])


#	command = "matlab -nodesktop -nosplash -r \"cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')\"" % (matlab_file_path, path_to_temp, path_to_temp)

#	call_silent.call_silent(os.system, command)
	os.system("stty echo")

		
	print ".......", time.time() - t, "sec"
	
	try:
		ws = call_silent.call_silent_err(spio.loadmat,"%s/watershed_result.mat" % path_to_temp)["ws"]
		os.system("rm %s/watershed_result.mat" % path_to_temp)
		os.system("rm %s/watershed_input.mat" % path_to_temp)
		os.system("rmdir %s" % path_to_temp)
	except IOError as err:
		err.message = "Could not read watershed result from Matlab. Perhaps Matlab did not run?"
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		sys.exit()
	return ws




	
def make_list_of_input_points(nuclei_collection, seed_collection = None):

	"""
	Prepare input points for watershed.
	"""

	len_nuclei = len(nuclei_collection.nuclei_list)
	try:
		len_seeds = len(seed_collection.list_of_seeds)
	except:
		len_seeds = 0
	init_pts = np.zeros((3, len_nuclei + len_seeds))
	
	counter = 0
	for nucleus in nuclei_collection.nuclei_list:
		init_pts[:,counter] = (nucleus.x, nucleus.y, nucleus.z)
		counter += 1
		
	if seed_collection:
		for seed in seed_collection.list_of_seeds:
			init_pts[:,counter] = (seed.x, seed.y, seed.z)
			counter += 1
		
	return init_pts
		
