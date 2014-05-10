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
	if len (nuclei_collection.nuclei_list):
		nucleus_index = nuclei_collection.nuclei_list[-1].index +1
	else:
		nucleus_index = 0
	new_nucleus = nc.Nucleus( asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin,  nucleus_index, added_by_user)
	
	nuclei_collection.add_nucleus (new_nucleus)



def get_nucleus_index_of_intended_label(segment_label, segment_collection, nuclei_collection):

	"""
	Given a segment label, what is the index of the nucleus associated with this segment.
	"""

	segment_list_index = segment_collection.segment_label_to_list_index_dict[segment_label]		
	segment = segment_collection.list_of_segments[segment_list_index]


	index = None

	for i in xrange(len(segment.nucleus_list)):
		nucleus_index_for_segment = segment.nucleus_list[0].index

		# return parent of this node, unless it's been deleted

		nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[nucleus_index_for_segment]
		parent_list_pos = nuclei_collection.union_find.find(nucleus_list_pos)

		if nuclei_collection.union_find.is_deleted[parent_list_pos]:
			break

		parent_nucleus = nuclei_collection.nuclei_list[parent_list_pos]

		index = parent_nucleus.index

	return index


def parse_to_delete_predictions(to_merge_predicted, segment_collection, nuclei_collection, incorrect_segments):


	for label1 in to_merge_predicted:

		nucleus_idx1 = get_nucleus_index_of_intended_label(label1, segment_collection, nuclei_collection)
	
		nucleus1 = nuclei_collection.nuclei_list[nucleus_idx1]
		nuclei_collection.delete_set_of_nucleus(nucleus1)

		idx1 = segment_collection.segment_label_to_list_index_dict[label1]
		seg1 = segment_collection.list_of_segments[idx1]

		incorrect_segments.add(seg1)
	

def parse_to_merge_predictions(to_merge_predicted, segment_collection, nuclei_collection, incorrect_segments):

	for pair in to_merge_predicted:
		label1 = pair[0]
		label2 = pair[1]
		
		nucleus_idx1 = get_nucleus_index_of_intended_label(label1, segment_collection, nuclei_collection)
		nucleus_idx2 = get_nucleus_index_of_intended_label(label2, segment_collection, nuclei_collection)
	
		nucleus1 = nuclei_collection.nuclei_list[nucleus_idx1]
		nucleus2 = nuclei_collection.nuclei_list[nucleus_idx2]			
		nuclei_collection.merge_two_nuclei(nucleus1, nucleus2)

		idx1 = segment_collection.segment_label_to_list_index_dict[label1]
		seg1 = segment_collection.list_of_segments[idx1]

		idx2 = segment_collection.segment_label_to_list_index_dict[label2]
		seg2 = segment_collection.list_of_segments[idx2]

		incorrect_segments.add(seg1)
		incorrect_segments.add(seg2)


def get_valid_segment_label_and_nucleus_index_from_user_click(right_click, box, label_map, segment_collection, nuclei_collection):

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
	
#	print segment_label , "@ %d, %d, %d in box %d, %d" % (asc_coords.xval, asc_coords.yval, asc_coords.zval, box.xmin, box.ymin )
#	test = np.zeros((label_map.shape[0], label_map.shape[1]))
#	test[box.xmin: box.xmax, box.ymin: box.ymax] = 1
#	pylab.imshow(test + np.double(label_map[:,:,box.zmin +asc_coords.zval] == segment_label))
#	pylab.show()


	if segment_label >1:
		# If the label selected is valid (no background/border), pick up the nucleus associated with it.		
		nucleus_index_for_segment = get_nucleus_index_of_intended_label(segment_label, segment_collection, nuclei_collection)
	else:
		return None, None
	
	return segment_label, nucleus_index_for_segment



def confirm_current_task_is_correct_and_apply(left_clicks, right_clicks, task_name, box, label_map, nuclei_collection, seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_segments):

	"""
	Given the left and right clicks, the label map and the task wanted,
	confirm that the user gave all the proper information for the requested task.
	"""


	def get_label_from_click(user_click):

		asc_coords = user_click.asc_coordinates
		label = label_map[ asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin]
		return label



	if task_name == "ADD_SEEDS_TO_NEW_LABEL":
		# if new label, then we need at least one left click. Take only the last left click.


		if len (left_clicks):
			user_mouse_click = left_clicks[-1]
			incorrect_labels.add(get_label_from_click(user_mouse_click))
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
			return False
			
		# check if no label
		if not (len(right_clicks)):
			message = "Ignoring ADD SEEDS TO EXISTING LABEL task. No label given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return False

		# get the last right click information
		segment_label, nucleus_index_for_segment = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-1], box, label_map, segment_collection, nuclei_collection)

		# make sure label is not blacklisted (marked for delete)
		if segment_label in blacklisted_segments:
			segment_label = None

		# if they returned None, None
		if not (segment_label and nucleus_index_for_segment):
			message = "Ignoring ADD SEEDS TO EXISTING LABEL task. Bad label (background, border or deleted)"
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return False


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
			incorrect_labels.add(get_label_from_click(user_mouse_click))
			incorrect_labels.add(segment_label)

				
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
			return False

		# check the last 2 right clicks:

		segment1, nucleus_index_for_segment1 = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-1], box, label_map, segment_collection, nuclei_collection)
		segment2, nucleus_index_for_segment2 = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-2], box, label_map, segment_collection, nuclei_collection)

		if (segment1 in blacklisted_segments) or (segment2 in blacklisted_segments):
			segment1 = None
			segment2 = None


		# if either one came None
		if not (segment1 and segment2 and nucleus_index_for_segment1 and nucleus_index_for_segment2):
			message = "Ignoring MERGE TWO LABELS task. Bad label given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return False

		# if we made it this far, merge the two labels (apply changes)
		nucleus1 = nuclei_collection.nuclei_list[nucleus_index_for_segment1]
		nucleus2 = nuclei_collection.nuclei_list[nucleus_index_for_segment2]			
		nuclei_collection.merge_two_nuclei(nucleus1, nucleus2)
		incorrect_labels.add(get_label_from_click(right_clicks[-2]))
		incorrect_labels.add(get_label_from_click(right_clicks[-1]))

	elif task_name == "ADD_BG_SEED":

		for user_mouse_click in left_clicks:

			coords = user_mouse_click.asc_coordinates
			bg_seed = (coords.xval+box.xmin, coords.yval+box.ymin, coords.zval+box.zmin)

			bg_seeds.add(bg_seed)



	return True


def confirm_delete_task_is_correct_and_add_label(left_clicks, right_clicks, task_name, box, label_map, nuclei_collection, seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_labels, nuclei_to_delete):


	"""
	Given the left and right clicks, the label map,
	confirm that the user gave all the proper information for the delete segment task.
	If so, add the label to be deleted to a set.
	"""


	def get_label_from_click(user_click):

		asc_coords = user_click.asc_coordinates
		label = label_map[ asc_coords.xval + box.xmin, asc_coords.yval + box.ymin, asc_coords.zval + box.zmin]
		return label

	

	if task_name == "DELETE_SEG":
		# if adding seed to an old label, check if 
		# (2) valid label is given (not background, not border)

		# check if no label
		if not (len(right_clicks)):
			message = "Ignoring DELETE SEGMENT task. No label given."
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return False

		# get the last right click information
		segment_label, nucleus_index_for_segment = get_valid_segment_label_and_nucleus_index_from_user_click(right_clicks[-1], box, label_map, segment_collection, nuclei_collection)

		# make sure label is not blacklisted (marked for delete)
		if segment_label in blacklisted_labels:
			segment_label = None

		# if they returned None, None
		if not (segment_label and nucleus_index_for_segment):
			message = "Ignoring DELETE SEGMENT task. Bad label (background or border)"
			logging.warning(message)
			print colored("Warning: %s" % message,"red")
			return False

		# add to blacklist, and list of nuclei to remove.

		blacklisted_labels.add(segment_label)
		nuclei_to_delete.add(nucleus_index_for_segment)


	return True
	


def get_blacklisted_segments(label_map, nuclei_collection, segment_collection, seed_collection, all_user_feedback, incorrect_segments, bg_seeds, blacklisted_labels):


	"""
	Go through all the user feedback and mark the segments to be deleted, 
	so that every interaction involving these segments can be ignored.
	"""

	task_index = -1
	current_task = ""

	task_left_click_buffer = []
	task_right_click_buffer = []

	box = None
	made_changes = False

	incorrect_labels = set()

	blacklisted_labels = set(blacklisted_labels)
	nuclei_to_delete = set()


	# get segments which need to be deleted

	for segment_gui_feedback in all_user_feedback:

	
		# for each click that was made in this segment correction window
		# what task did that click associated with it?
		# did the user provide all the information?
		
		# detect all the tasks that the user wanted
		# make changes for the ones with complete information



		for user_mouse_click in segment_gui_feedback.list_of_cropped_ascidian_events:
		
			# task index starts at -1. Reinitialized to the index of current task from clicks.
			# Also check if current task is legit (has all info), otherwise, discard it.
			

			if user_mouse_click.task_index != task_index:
				# Finish business with current task                              
				if current_task == "DELETE_SEG" and box:
					made_changes = confirm_delete_task_is_correct_and_add_label(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection, seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_labels, nuclei_to_delete) or made_changes 
				

				# Initialize next task
				task_index = user_mouse_click.task_index
				current_task = user_mouse_click.button_task

				# empty click buffers per task
				task_left_click_buffer = []
				task_right_click_buffer = []
				box = segment_gui_feedback.bounding_box
	
			if user_mouse_click.right_click:
				task_right_click_buffer.append(user_mouse_click)
			else:
				task_left_click_buffer.append(user_mouse_click)

	if current_task == "DELETE_SEG" and box:
		made_changes = confirm_delete_task_is_correct_and_add_label(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection,seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_labels, nuclei_to_delete) or made_changes
				

	return blacklisted_labels, nuclei_to_delete, made_changes




def parse_user_feedback(label_map, nuclei_collection, segment_collection, seed_collection, all_user_feedback, incorrect_segments, bg_seeds, to_delete_predicted):

	"""
	Given all the user feedback, extract relevant information and make changes accordingly:
	- add nuclei, add seeds, modify union of nuclei, etc.
	"""


	task_index = -1
	current_task = ""

	task_left_click_buffer = []
	task_right_click_buffer = []

	box = None
	made_changes = False

	incorrect_labels = set()

	blacklisted_segments = set(to_delete_predicted)
	blacklisted_segments, nuclei_to_delete, made_changes = get_blacklisted_segments(label_map, nuclei_collection, segment_collection, seed_collection, all_user_feedback, incorrect_segments, bg_seeds, blacklisted_segments)

	# for each segment correction window that was open:
	for segment_gui_feedback in all_user_feedback:

		
	
		# for each click that was made in this segment correction window
		# what task did that click associated with it?
		# did the user provide all the information?
		
		# detect all the tasks that the user wanted
		# make changes for the ones with complete information



		for user_mouse_click in segment_gui_feedback.list_of_cropped_ascidian_events:
		
			# task index starts at -1. Reinitialized to the index of current task from clicks.
			# Also check if current task is legit (has all info), otherwise, discard it.
			

			if user_mouse_click.task_index != task_index:
				# Finish business with current task                              
				if current_task != "NO_TASK_SELECTED" and box:
					made_changes = confirm_current_task_is_correct_and_apply(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection, seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_segments) or made_changes 
				

				# Initialize next task
				task_index = user_mouse_click.task_index
				current_task = user_mouse_click.button_task

				# empty click buffers per task
				task_left_click_buffer = []
				task_right_click_buffer = []
				box = segment_gui_feedback.bounding_box
	
			if user_mouse_click.right_click:
				task_right_click_buffer.append(user_mouse_click)
			else:
				task_left_click_buffer.append(user_mouse_click)

	if current_task != "NO_TASK_SELECTED" and box:
		made_changes = confirm_current_task_is_correct_and_apply(task_left_click_buffer, task_right_click_buffer, current_task, box, label_map, nuclei_collection,seed_collection, segment_collection, incorrect_labels, bg_seeds, blacklisted_segments) or made_changes
				

	# take all the incorrect labels and get the segments corresponding


	for label in incorrect_labels:

		if label >1:
			idx = segment_collection.segment_label_to_list_index_dict[label]
			incorrect_segments.add(segment_collection.list_of_segments[idx])


	# delete all the nuclei to delete..

	for nucleus_index in nuclei_to_delete:

		nuclei_collection.delete_set_of_nucleus_by_index(nucleus_index)

	

	return made_changes







