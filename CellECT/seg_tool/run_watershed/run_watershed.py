# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import os
from scipy import io as spio
import time
import numpy as np


import CellECT.seg_tool.globals

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
	
	print "\nRunning seeded watershed....\n"
	has_bg = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["has_bg"])
	spio.savemat("temp/watershed_input.mat", {"vol":vol, "seeds": init_pts, "has_bg": has_bg})
	import subprocess
	import os
	t = time.time()
	
	os.system( "matlab -nodesktop -nosplash -r \"cd utils; run_seeded_watershed('../temp/watershed_input.mat', '../temp/watershed_result.mat')\"")
	os.system("stty echo")

		
	print ".......", time.time() - t, "sec"
	
	ws = spio.loadmat("temp/watershed_result.mat")["ws"]
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
		
