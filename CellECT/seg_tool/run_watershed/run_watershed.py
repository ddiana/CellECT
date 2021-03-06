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
import os.path

# Imports from this project
import CellECT.seg_tool.globals
from CellECT.seg_tool.seg_utils import call_silent


"""
This module prepares input and calls watershed from Matlab.
TODO: Rplace this with custom C-extension of watershed algorithm to remove the
need for matlab.
"""

def run_watershed(vol, init_pts, bg_seeds, bg_mask):

	"""
	This function calls matlab to run watershed on the given volume.
	TODO: Replace this with C extension of watershed.
	"""
	

	if bg_mask is None:
		bg_mask = []

	path_to_temp = tempfile.mkdtemp()



	print "Running seeded watershed...."
	has_bg = int(CellECT.seg_tool.globals.DEFAULT_PARAMETER["has_bg"])

	save_mat_file = os.path.join(path_to_temp, "watershed_input.mat")

	sbx = [ x[0] for x in bg_seeds ]
	sby = [ x[1] for x in bg_seeds ]
	sbz = [ x[2] for x in bg_seeds ]

	# add dummy values in the list to make sure the values are ported in the right format (wtf scipy io?!)

	init_pts.append([[-1,-1,-1]])
	init_pts.append([[-1,-1,-1], [-1,-1,-1]])

	print path_to_temp


	
	try:
		call_silent.call_silent_err(spio.savemat, save_mat_file, {"vol":vol, "seeds": init_pts, "has_bg": has_bg, "sbx": sbx, "sby": sby, "sbz": sbz, "bg_mask": bg_mask})
	except Exception as err:
		err.message = "Could not write input file for Matlab at %s" % save_mat_file
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		sys.exit()


	t = time.time()
	
	
	matlab_file_path = os.path.join(CellECT.__path__[0] , "utils")	


	with open(os.devnull, "wb") as devnull:
		subprocess.check_call( ["matlab", "-nodesktop", "-nosplash", "-r", "cd %s; run_seeded_watershed('%s', '%s')" % (os.path.join(matlab_file_path), os.path.join(path_to_temp, "watershed_input.mat"), os.path.join(path_to_temp, "watershed_result.mat"))], stdout=devnull, stderr=subprocess.STDOUT)


#			if len(bg_seeds) >1:
#
#				mask = (label_map == 1)
#				mask = ndimage.binary_dilation( mask)
#				mask = ndimage.binary_dilation( mask)
#				mask = ndimage.binary_erosion( mask)
#				mask = ndimage.binary_erosion( mask)
#
#				mask = (label_map > 1)
#				mask = ndimage.binary_dilation( mask)
#				mask = ndimage.binary_dilation( mask)

#				label_map = label_map * mask + (1-mask)
				
##				label_map = mask + (1-mask) * (label_map)

##				label_map[:,:,0] = 0
##				label_map[:,:,-1] = 0
##				label_map[:,0,:] = 0
##				label_map[:,-1,:] =0
##				label_map[0,:,:] = 0
##				label_map[-1,:,:] = 0
##				label_map[:,:,1] = 0
##				label_map[:,:,-2] = 0
##				label_map[:,1,:] = 0
##				label_map[:,-2,:] =0
##				label_map[1,:,:] = 0
##				label_map[-2,:,:] = 0


#	subprocess.check_call( ["matlab", "-nodesktop", "-nosplash", "-r", "cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')" % (matlab_file_path, path_to_temp, path_to_temp)])


#	command = "matlab -nodesktop -nosplash -r \"cd %s; run_seeded_watershed('%s/watershed_input.mat', '%s/watershed_result.mat')\"" % (matlab_file_path, path_to_temp, path_to_temp)

#	call_silent.call_silent(os.system, command)
	os.system("stty echo")

	print ".......", time.time() - t, "sec"
	
	try:
		ws = call_silent.call_silent_err(spio.loadmat, os.path.join(path_to_temp,"watershed_result.mat"))["ws"]
		os.system("rm %s" % os.path.join(path_to_temp,"watershed_result.mat"))
		os.system("rm %s" % os.path.join(path_to_temp, "watershed_input.mat"))
		os.system("rmdir %s" % path_to_temp)
	except IOError as err:
		err.message = "Could not read watershed result from Matlab. Perhaps Matlab did not run?"
		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		sys.exit()
	return ws



def make_list_of_seed_groups(nuclei_collection, seed_collection = None):

	dict_of_grouped_seeds = {}


	for nucleus in nuclei_collection.list_all_nuclei():
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
			if	head_nucleus is not None:
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
	for nucleus in nuclei_collection.list_all_nuclei():
		init_pts[:,counter] = (nucleus.x, nucleus.y, nucleus.z)
		counter += 1
		
	if seed_collection:
		for seed in seed_collection.list_of_seeds:
			init_pts[:,counter] = (seed.x, seed.y, seed.z)
			counter += 1
		
	return init_pts
		
