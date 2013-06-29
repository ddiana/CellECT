# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import os
from scipy import io as spio
import time
import numpy as np
import pdb

import CellECT.seg_tool.globals
import CellECT

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

	path_to_temp = CellECT.seg_tool.globals.path_to_workspace + "/temp/"

	## Make temp folder ########################################################
	if not os.path.exists(path_to_temp):
		os.makedirs(path_to_temp)

	
	print "\nRunning seeded watershed....\n"
	has_bg = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["has_bg"])
	spio.savemat("%s/watershed_input.mat" % path_to_temp, {"vol":vol, "seeds": init_pts, "has_bg": has_bg})
	import subprocess

	t = time.time()
	
	matlab_file_path = CellECT.__path__[0] + "/utils"

	
	os.system( "matlab -nodesktop -nosplash -r \"cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')\"" % (matlab_file_path, path_to_temp, path_to_temp))
	os.system("stty echo")

		
	print ".......", time.time() - t, "sec"
	
	ws = spio.loadmat("%s/watershed_result.mat" % path_to_temp)["ws"]
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
		
