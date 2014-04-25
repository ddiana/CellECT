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

def run_watershed(vol, init_pts, bg_seeds):

	"""
	This function calls matlab to run watershed on the given volume.
	TODO: Replace this with C extension of watershed.
	"""

	path_to_temp = tempfile.mkdtemp()


	
	print "Running seeded watershed...."
	has_bg = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["has_bg"])

	save_mat_file = "%s/watershed_input.mat" % path_to_temp

	bg_seeds_temp = [ list (x) for x in bg_seeds ]



	try:
		call_silent.call_silent_err(spio.savemat, save_mat_file, {"vol":vol, "seeds": init_pts, "has_bg": has_bg, "background_seeds": bg_seeds_temp})
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



def make_list_of_seed_groups(nuclei_collection, seed_collection = None):

	dict_of_grouped_seeds = {}

	
	for nucleus in nuclei_collection.nuclei_list:
		head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(nucleus)
		if dict_of_grouped_seeds.has_key(head_nucleus.index):
			dict_of_grouped_seeds[head_nucleus.index].append([nucleus.x, nucleus.y, nucleus.z])
		else:
			dict_of_grouped_seeds[head_nucleus.index] = [[nucleus.x, nucleus.y, nucleus.z]]


	if seed_collection is not None:
		for seed in seed_collection.list_of_seeds:
			index_of_parent_nucleus = seed.nucleus_index
			# get the nucleus object
			parent_nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[index_of_parent_nucleus]
			parent_nucleus = nuclei_collection.nuclei_list[parent_nucleus_list_pos]
			# get the head nucleus of the set this nucleus is (in case it was merged with soemthing else)
			head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(parent_nucleus)		
			dict_of_grouped_seeds[head_nucleus.index].append([seed.x, seed.y, seed.z])



	return dict_of_grouped_seeds.values()


	
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
		
