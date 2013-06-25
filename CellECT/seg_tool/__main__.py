# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Module information
import module_info

# Imports
import pdb
import numpy as np
import scipy.io as spio
from PyML import VectorDataSet
import sys
from termcolor import colored
import os
import time


# Imports from this project
from seg_io import load_parameters as loader
from cellness_metric import cellness_classifier as cellness
from seg_io import load_all
from nuclei_collection import nuclei_collection as nc
from seed_segment_collection import seed_segment_collection as ssc
from seed_collection import seed_collection as seedc
from features import segment_features as feat
from seg_utils import call_silent
from gui import main_gui
from gui import parse_feedback as gui_parse
from seg_io import save_all
from seg_utils import display_tools
from seg_utils import labelmap_tools as lm_tools
from run_watershed import run_watershed as ws

import globals


"""
Main sequence of events of the seg_tool, the interactive segmentation tool 
from the CellECT (Cell Evolution Capturing Tool) project.
"""
	
def main():

	## CHECK INPUT #############################################################
	if len(sys.argv) != 2:
		print module_info.__usage__
		return
	else:
		config_file_path = sys.argv[1]


	## Make temp folder ########################################################
	if not os.path.exists("temp"):
		os.makedirs("temp")

	## READ AND SET PARAMETERS #################################################
	
	print "Loading config parameters..."
	loader.read_program_parameters(config_file_path)



	## LOADING TRAINING DATA AND TRAINING CLASSIFIER ###########################

	print colored("================================================================================",'red')
	print colored("              Loading training data and training cellness metric:", 'red')
	print colored("================================================================================",'red')
	
	classifier = cellness.learn_classifier()


	## LOADING DATA FOR CURRENT VOLUME #########################################
	
	print colored("================================================================================",'red')
	print colored("                            Processing test volume:",'red')
	print colored("================================================================================",'red')
	
	print "Loading test volume..."
	vol = load_all.load_from_mat(globals.DEFAULT_PARAMETER["volume_mat_path"], globals.DEFAULT_PARAMETER["volume_mat_var"])
	print "Loading initial segmentation..."
	watershed = load_all.load_from_mat(globals.DEFAULT_PARAMETER["first_seg_mat_path"], globals.DEFAULT_PARAMETER["first_seg_mat_var"])

	#watershed = shuffle_labels(watershed)

	
	print "Loading nuclei..."
	nuclei_collection = nc.NucleusCollection(globals.DEFAULT_PARAMETER["nuclei_mat_path"], globals.DEFAULT_PARAMETER["nuclei_mat_var"])

	last_length_of_seeds_list = 0 
	seed_collection = seedc.SeedCollection([])
	seed_segment_collection = ssc.SeedSegmentCollection()


	ask_feedback = True
	watershed_old = watershed

	globals.should_load_last_save
	globals.should_load_last_save = False

	
	## MAIN LOOP ###############################################################

	while ask_feedback or globals.should_load_last_save:

		## Load last save, or continue with current setup ######################
		if globals.should_load_last_save:
			nuclei_collection, seed_collection, watershed = load_all.load_last_save()
			seed_segment_collection.update_seed_segment_collection(seed_segment_collection, watershed, seed_collection)
			#watershed = recolor_label_map_correctly (watershed, nuclei_collection, seed_collection, collection_of_ws_segments, seed_segment_collection)
	
			globals.should_load_last_save = False


		## Prepare cellness metric #############################################

		print "Preparing data for classifier..."

		set_of_labels = set(int(x) for x in np.unique(watershed) if x > 1)	
		original_init_pts = ws.make_list_of_input_points(nuclei_collection)
		collection_of_ws_segments = feat.get_segments_with_features(vol, watershed, set_of_labels, "watershed", nuclei_collection)

		test_data = cellness.prepare_test_data(collection_of_ws_segments)
		test_data_svm = call_silent.call_silent_process(VectorDataSet, test_data)

		## Apply cellness metric ###############################################

		print "Applying cellness metric to segments..."
		class_prediction, discriminant_value = cellness.classify_segments(classifier,test_data_svm)

		classified_segments = (set_of_labels, class_prediction, discriminant_value)


		## ASK FOR FEEDBACK ####################################################

		print "Prompting user for feedback..."	
		all_user_feedback = main_gui.show_uncertainty_map_and_get_feedback( vol, watershed, collection_of_ws_segments, classified_segments, nuclei_collection, seed_collection, seed_segment_collection, watershed_old)
	

		## USE FEEDBACK ########################################################

		if not globals.should_load_last_save:

			print "Processing user feedback..."
			old_number_of_nuclei = len(nuclei_collection.nuclei_list)
			old_number_of_seeds = len(seed_collection.list_of_seeds)
		
			## Get feedback ####################################################
			gui_parse.parse_user_feedback(watershed, nuclei_collection, collection_of_ws_segments, seed_collection, all_user_feedback)	
		
			new_number_of_nuclei = len(nuclei_collection.nuclei_list)
			new_number_of_seeds = len(seed_collection.list_of_seeds)

			## Apply Feedback ##################################################
	
			if len(all_user_feedback):

				init_pts = ws.make_list_of_input_points(nuclei_collection, seed_collection)
				#call_silent_err( show_new_user_points,original_init_pts, init_pts)

				watershed_old = watershed

				## Rerun Watershed if necessary ################################
				if old_number_of_nuclei != new_number_of_nuclei or old_number_of_seeds != new_number_of_seeds:
				
					watershed = ws.run_watershed(vol, init_pts)
					seed_segment_collection.update_seed_segment_collection(seed_segment_collection, watershed, seed_collection)


				watershed = lm_tools.recolor_label_map_correctly (watershed, nuclei_collection, seed_collection, collection_of_ws_segments, seed_segment_collection)

				del test_data
				del test_data_svm
				del classified_segments
				del all_user_feedback
				del init_pts
			
			else:
				ask_feedback = False


			
	## Show final output #######################################################
	display_tools.display_volume_two(vol, watershed)


	## Ask if save #############################################################
	should_save = ""
	while not should_save in set(['y', 'n']):
		print colored("Save latest result? [Y/N] ","red")
		should_save = sys.stdin.read(1)
		should_save = should_save.lower()
		print ""

		
	## Save ####################################################################
	if should_save.lower() == "y":
		save_all.save_current_status(nuclei_collection, seed_collection, collection_of_ws_segments, seed_segment_collection, watershed)
	else:
		print "Not saving."
	

	print "KTnxBye."




if __name__ == "__main__":


	main()



