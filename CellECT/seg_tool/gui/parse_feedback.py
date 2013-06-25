# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import pylab
import time
from termcolor import colored

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
	

def get_nucleus_index_of_intended_label(segment_gui_feedback, task_of_interest, label_map, segment_collection):
	# if there was a right click, get nucleus it was meant to pick up

	"""
	When the user right-clicked on a label, what nucleus is associated with that label.
	Particullarly, if this is part of a union of segments, which one is the head nucleus?
	"""


	nucleus_index = []
	counter = 0

	# find where the task of interest clicks begin:
	while segment_gui_feedback.list_of_cropped_ascidian_events[counter].task_index != task_of_interest:
		counter += 1
	task_index = segment_gui_feedback.list_of_cropped_ascidian_events[counter].task_index
	

	box = segment_gui_feedback.bounding_box
	while task_index == task_of_interest and counter < len(segment_gui_feedback.list_of_cropped_ascidian_events):
	
		user_mouse_click = segment_gui_feedback.list_of_cropped_ascidian_events [counter]
		
		# if right click - if this click was intended to pick up a segment label
		if segment_gui_feedback.list_of_cropped_ascidian_events[counter].right_click:
			asc_coords = user_mouse_click.asc_coordinates

			segment_label =  label_map[ box.xmin + asc_coords.xval, box.ymin + asc_coords.yval, box.zmin + asc_coords.zval  ]
		
			# get the nucleus associated with this segment.
			try:
				segment_list_index = segment_collection.segment_label_to_list_index_dict[segment_label]
				segment = segment_collection.list_of_segments[segment_list_index]
				nucleus_index.append(segment.nucleus_list[0].index)

			except:
				print "Warning: Segment with label", segment_label,"does not exist. Perhaps background/border."
				
			
			
		counter += 1
		try:
			task_index = segment_gui_feedback.list_of_cropped_ascidian_events[counter].task_index
		except:
			break
		

	return nucleus_index
	

	

def parse_user_feedback(label_map, nuclei_collection, segment_collection, seed_collection, all_user_feedback):

	"""
	Given all the user feedback, extract relevant information and make changes accordingly:
	- add nuclei, add seeds, modify union of nuclei, etc.
	"""

	task_index = -1
	
	no_task = True
	new_nucleus = nc.Nucleus (-1,-1,-1,-1)

	
	# for list in list of lists
	for segment_gui_feedback in all_user_feedback:
		# for mouse event in list (a list for every segment gui)
		box = segment_gui_feedback.bounding_box
		
		for user_mouse_click in segment_gui_feedback.list_of_cropped_ascidian_events:
		
			if user_mouse_click.task_index != task_index:
			# if we found new task, initialize each task accordingly
				task_index = user_mouse_click.task_index
#				print "task_index", task_index
#				print "task:", user_mouse_click.button_task
				
				# what is that new task?
				if user_mouse_click.button_task == "NO_TASK_SELECTED":
				 	no_task = True
				 	continue
				 
				elif user_mouse_click.button_task == "ADD_SEEDS_TO_EXISTING_LABEL":
			
					no_task = False
					# get the nucleus corresponding to the latest right mouse clicked segment				
					nucleus_index_list = get_nucleus_index_of_intended_label(segment_gui_feedback,  task_index, label_map, segment_collection) 
			
					if len(nucleus_index_list) >0:					
						nucleus_index = nucleus_index_list[-1]
					else:
						print  "Warning: Cannnot perform add seed to old label task. Bad or no label."

	
				elif user_mouse_click.button_task == "ADD_SEEDS_TO_NEW_LABEL":
				
					no_task = False
					# make a new nucleus at the position of this seed:
					nucleus_index = make_new_nucleus_and_return_index(user_mouse_click, box,  nuclei_collection, added_by_user=True)
					
				elif user_mouse_click.button_task == "MERGE_TWO_LABELS":
			
					no_task = False
					# get nucleus index of each intended label (last 2 right clicks):
					#print "task index", task_index
					nucleus_index_list = get_nucleus_index_of_intended_label(segment_gui_feedback, task_index, label_map, segment_collection)
					#print nucleus_index_list		
	
					if len(nucleus_index_list) >= 2:
						nucleus_index1 = nucleus_index_list[-1]
						nucleus_index2 = nucleus_index_list[-2]
	
						nucleus1 = nuclei_collection.nuclei_list[nucleus_index1]
						nucleus2 = nuclei_collection.nuclei_list[nucleus_index2]
					
						nuclei_collection.merge_two_nuclei(nucleus1, nucleus2)
					else:
						print "Warning: Cannot perform merge labels task. Bad or no labels."
					
				else:
				
					print "GOT BAD BUTTON TASK"
					pdb.set_trace()
					
			else:   # if not a new task
				
				# what task were were working on?
				
				if user_mouse_click.button_task == "NO_TASK_SELECTED":
				 	continue   # skip to next
				 
				elif user_mouse_click.button_task == "ADD_SEEDS_TO_EXISTING_LABEL":
					# these are both the same..
					# because seeds to new nucleus already had a new nucleus created. so every after it, gets attached to that nucleus index
				
					# increment the index of the largest seed
					try: 
						seed_index = seed_collection.list_of_seeds[-1].index +1
					except:
						seed_index = 0
					new_seed = make_new_seed(user_mouse_click, box, nucleus_index, seed_index )
					seed_collection.add_seed(new_seed)

				elif user_mouse_click.button_task == "ADD_SEEDS_TO_NEW_LABEL":
					pass
					
				elif user_mouse_click.button_task == "MERGE_TWO_LABELS":
				
					pass
					# TODO
				else:
				
					print "========================================GOT BAD BUTTON TASK"
					pdb.set_trace()
					

