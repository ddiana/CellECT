# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import pylab
import time
from termcolor import colored
import logging

# Imports from this project
from CellECT.seg_tool.nuclei_collection import nuclei_collection as nc
from CellECT.seg_tool.seed_collection import seed_collection as seedc


"""
Get the meaningful information from the user clicks. What did the user intend to do?
"""

def get_user_corrections( vol, label_map, collection_of_segments, classification, set_of_labels ):
	"""
	Make a list of the mouse events from every segment correction GUI call.
	"""

	mouse_events = []

	counter = 1
	
	total = len(np.nonzero(classification))
	collection_of_segments.update_index_dict()
	
	for index in len(np.nonzero(classification)):
	
		print "------------- Segment", counter, "of", total, "---------------"
	
		label = set_of_labels[index]
		list_index = collection_of_segments.segment_label_to_list_index_dict[label]
		
		(s, m1, m2) = display_segment_to_correct(vol, label_map, collection_of_segments.list_of_segments[list_index])
		
		mouse_events.append((collection_of_segments.list_of_segments[list_index].label, (s,m1,m2)))
		counter+=1
		
	return mouse_events
	

def make_new_nucleus_and_return_index(user_mouse_click,box, nuclei_collection, added_by_user = False):

	"""
	Make a new nucleus object.
	"""

	asc_coords = user_mouse_click.asc_coordinates
	
	nucleus_index = nuclei_collection.nuclei_list[-1].index +1
	new_nucleus = nc.Nucleus( asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin,  nucleus_index, added_by_user)
	
	nuclei_collection.add_nucleus (new_nucleus)
	



def make_new_seed(user_mouse_click, box, nucleus_index, seed_index):
	
	"""
	Make new seed object.
	"""

	asc_coords = user_mouse_click.asc_coordinates
	seed = seedc.Seed(asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin, nucleus_index, seed_index)
	
	return seed
	


def add_new_nucleus_to_collection(user_mouse_click,box, nuclei_collection, added_by_user = True ):

	"""
	Take user click in segment window, adjust to global window, and add nucleus with those coordinates to the collection.
	"""

	asc_coords = user_mouse_click.asc_coordinates
	
	nucleus_index = nuclei_collection.nuclei_list[-1].index +1
	new_nucleus = nc.Nucleus( asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin,  nucleus_index, added_by_user)
	
	nuclei_collection.add_nucleus (new_nucleus)



def get_nucleus_index_of_intended_label(segment_label, segment_collection):

	"""
	Given a segment label, what is the index of the nucleus associated with this segment.
	"""

	segment_list_index = segment_collection.segment_label_to_list_index_dict[segment_label]		
	segment = segment_collection.list_of_segments[segment_list_index]
	nucleus_index_for_segment = segment.nucleus_list[0].index

	return nucleus_index_for_segment



def get_valid_segment_label_and_nucleus_index_from_user_click(right_click, box, label_map, segment_collection):

	"""
	Given a user right click, what was the segment label associated with it?
	WHat was the nucleus associated with this segment?
	If the segment label is invalid (0,1), return None
	"""
	# check that the label is valid (not background - 1, not border - 0)
	# only look at the last right click in this sequence

	asc_coords = right_click.asc_coordinates
	segment_label =  label_map[ box.xmin + asc_coords.xval, box.ymin + asc_coords.yval, box.zmin + asc_coords.zval  ]
	nucleus_index_for_segment = -1

	if segment_label >1:
		# If the label selected is valid (no background/border), pick up the nucleus associated with it.		
		nucleus_index_for_segment = get_nucleus_index_of_intended_label(segment_label, segment_collection)
	else:
		return None, None
	
	return segment_label, nucleus_index_for_segment


def confirm_current_task_is_correct_and_apply(left_clicks, right_clicks, task_name, box, label_map, nuclei_collection, seed_collection, segment_collection):

	"""
	Given the left and right clicks, the label map and the task wanted,
	confirm that the user gave all the proper information for the requested task.
	"""

	if task_name == "ADD_SEEDS_TO_NEW_LABEL":
		# if new label, then we need at least one left click. Take only the last left click.


		if len (left_clicks):
			user_mouse_click = left_clicks[-1]
			nucleus_index = add_new_nucleus_to_collection(user_mouse_click, box,  nuclei_collection, added_by_user=True)
		else:
			message = "Ignoring ADD SEED TO NEW LABEL task. No seed given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")

	elif task_name == "ADD_SEEDS_TO_EXISTING_LABEL":
		# if adding seed to an old label, check if 
		# (1) at least one seed is given, 
		# (2) valid label is given (not background, not border)

		# check if no seed
		if not (len(left_clicks)):
			message = "Ignoring ADD SEED TO NEW LABEL task. No seed given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return
			
		# check if no label
		if not (len(right_clicks)):
			message = "Ignoring ADD SEEDS TO EXISTING LABEL task. No label given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return

		# get the last right click information
		segment_label, nucleus_index_for_segment = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-1], box, label_map, segment_collection)

		# if they returned None, None
		if not (segment_label and nucleus_index_for_segment):
			message = "Ignoring ADD SEEDS TO EXISTING LABEL task. Bad label (background or border)"
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return


		# If we made it so far, that means we have a good label, and a nonzero list of seeds.
		# Add the seeds to the new label. (Apply changes)

		for user_mouse_click in left_clicks:

			# increment the index of the largest seed
			try: 
				seed_index = seed_collection.list_of_seeds[-1].index +1
			except:
				seed_index = 0
			# make the new seed and pair it with its nucleus.
			new_seed = make_new_seed(user_mouse_click, box, nucleus_index_for_segment, seed_index )
			# add the new seed to the collection.
			seed_collection.add_seed(new_seed)

				
	elif task_name == "MERGE_TWO_LABELS":
		# Make sure the user gave two valid label selections:
		# two right clicks, which:
		# 1) are distinct
		# 2) neither is border
		# 3) neither is background.

		if len(right_clicks) <2:
			message = "Ignoring MERGE TWO LABELS task. Not enough labels given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return

		# check the last 2 right clicks:

		segment1, nucleus_index_for_segment1 = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-1], box, label_map, segment_collection)
		segment2, nucleus_index_for_segment2 = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-2], box, label_map, segment_collection)

		# if either one came None
		if not (segment1 and segment2 and nucleus_index_for_segment1 and nucleus_index_for_segment2):
			message = "Ignoring MERGE TWO LABELS task. Bad label given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return

		# if we made it this far, merge the two labels (apply changes)
		nucleus1 = nuclei_collection.nuclei_list[nucleus_index_for_segment1]
		nucleus2 = nuclei_collection.nuclei_list[nucleus_index_for_segment2]			
		nuclei_collection.merge_two_nuclei(nucleus1, nucleus2)





def parse_user_feedback(label_map, nuclei_collection, segment_collection, seed_collection, all_user_feedback):

	"""
	Given all the user feedback, extract relevant information and make changes accordingly:
	- add nuclei, add seeds, modify union of nuclei, etc.
	"""

	task_index = -1
	current_task = ""

	task_left_click_buffer = []
	task_right_click_buffer = []

	box = None

	# for each segment correction window that was open:
	for segment_gui_feedback in all_user_feedback:
	
		# for each click that was made in this segment correction window
		# what task did that click associated with it?
		# did the user provide all the information?
		
		# detect all the tasks that the user wanted
		# make changes for the ones with complete information

		box = segment_gui_feedback.bounding_box

		for user_mouse_click in segment_gui_feedback.list_of_cropped_ascidian_events:
		
			# task index starts at -1. Reinitialized to the index of current task from clicks.
			# Also check if current task is legit (has all info), otherwise, discard it.

			if user_mouse_click.task_index != task_index:
				# Finish business with current task                              
				if current_task != "NO_TASK_SELECTED":
					confirm_current_task_is_correct_and_apply(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection, seed_collection, segment_collection)
				

				# Initialize next task
				task_index = user_mouse_click.task_index
				current_task = user_mouse_click.button_task

				# empty click buffers per task
				task_left_click_buffer = []
				task_right_click_buffer = []
	
			if user_mouse_click.right_click:
				task_right_click_buffer.append(user_mouse_click)
			else:
				task_left_click_buffer.append(user_mouse_click)

	if current_task != "NO_TASK_SELECTED" and box:
		confirm_current_task_is_correct_and_apply(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection,seed_collection, segment_collection)
				








