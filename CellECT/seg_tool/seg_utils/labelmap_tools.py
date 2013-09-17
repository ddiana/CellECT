# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
from numpy import random
import pdb
import copy
import time


"""
Label map related utilities:
	- shuffle labels
	- recolor the label map according to head nucleus assignment
	- remove boundaries for segments that have the same label.
"""


def shuffle_labels(label_map):

	"Randomize labels in a label map."


	# TODO write this with iterators

	t = time.time()

	labels = np.unique(label_map)
	labels = sorted(labels)

	old_labels = copy.deepcopy(labels)
	old_labels_dict = dict((label, index) for index, label in enumerate(old_labels))

	np.random.seed(12)
	np.random.shuffle(labels)

	for i in xrange(len(labels)):
		if labels[i] == 0:
			temp = labels[i]
			labels[i] = labels[0]
			labels[0] = temp

		if labels[i] == 1:
			temp = labels[i]
			labels[i] = labels[1]
			labels[1] = temp

		if labels[0] == 0 and labels[1] == 1:
			break

#		reverse_index = {label: [] for label in set_of_labels}

#		it = np.nditer(label_map, flags=['multi_index'])
#		while not it.finished:
#			label = int(it[0])
#			if label in set_of_labels:
#				reverse_index[label].append(it.multi_index)
#			it.iternext()
	
	it = np.nditer(label_map, op_flags=['readwrite'])
	for x in np.nditer(label_map, op_flags =["readwrite"]):

		label = int(x)
		x[...] =  labels[old_labels_dict[label]]
	

#	for i in xrange(label_map.shape[0]):
#		for j in xrange(label_map.shape[1]):
#			for k in xrange(label_map.shape[2]):

#				label_map[i,j,k] = labels[old_labels_dict[label_map[i,j,k]]]

	print "label scramble:" , time.time() - t

	return label_map




def recolor_label_map_correctly( label_map, nuclei_collection, seed_collection, segment_collection, seed_segment_collection):
	"Reassign labels according to head nucleus of every segment (or seed segment)"

	# TODO: only do these changes locally, dont do operations on the whole matrix

	set_of_labels_for_which_to_remove_border = set()

	# check if already colored correctly (in case user said just merge 2 nuclei, and watershed was not rerun)

	# color all the seed segment by their head nucleus:
	for seed in seed_collection.list_of_seeds:
	
		# curent label in the label map (comes from watershed)
		current_label = label_map[seed.x, seed.y, seed.z]

		if current_label >1:

			# get the nucleus index this seed is associated with (if any)
			index_of_parent_nucleus = seed.nucleus_index
			
			# get the nucleus object
			parent_nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[index_of_parent_nucleus]
			parent_nucleus = nuclei_collection.nuclei_list[parent_nucleus_list_pos]
			# get the head nucleus of the set this nucleus is (in case it was merged with soemthing else)
			head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(parent_nucleus)
		
			# color it by the current color of the head nucleus:
			new_label = label_map[head_nucleus.x, head_nucleus.y, head_nucleus.z]
			
			if current_label != new_label:
				#print current_label ," => ", new_label
				label_map = (label_map == current_label) * new_label + (label_map != current_label) * label_map


				# take the box around this seed segment and remove boundary if necessary
				try:
					seed_segment_list_index = seed_segment_collection.seed_index_to_seed_segment_list_index_dict[seed.index]
				except:
					pdb.set_trace()
				seed_segment = seed_segment_collection.list_of_seed_segments[seed_segment_list_index]
				bb = seed_segment.bounding_box
				remove_boundary(label_map[bb.xmin:bb.xmax, bb.ymin:bb.ymax, bb.zmin:bb.zmax])

			

	# check if already colored correctly (in case user said just merge 2 nuclei, and watershed was not rerun)

	# color every nucleus segment to the color of the head nucleus.
	for nucleus in nuclei_collection.nuclei_list:
		head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(nucleus)

		if head_nucleus.index != nucleus.index:

			current_label = label_map[nucleus.x, nucleus.y, nucleus.z]
			new_label = label_map[head_nucleus.x, head_nucleus.y, head_nucleus.z]

			if current_label != new_label:

				# got the label that this index should have
				# TODO: dont iterate the whole matrix, only redo the the pixels in that segment
				label_map = (label_map == current_label) * new_label + (label_map != current_label) * label_map

				# remove boundary, if any of the nuclei to be merged exist as segments
				try:
					segment_index_1 = segment_collection.segment_label_to_list_index_dict[current_label]
					segment1 = segment_collection.list_of_segments[segment_index_1]
					bb1 = segment1.bounding_box
					remove_boundary(label_map[bb1.xmin:bb1.xmax, bb1.ymin:bb1.ymax, bb1.zmin:bb1.zmax])
				except:
					try:
						segment_index_2 = segment_collection.segment_label_to_list_index_dict[new_label]
						segment2 = segment_collection.list_of_segments[segment_index_2]			
						bb2 = segment2.bounding_box
						remove_boundary(label_map[bb2.xmin:bb2.xmax, bb2.ymin:bb2.ymax, bb2.zmin:bb2.zmax])
					except:
						pass
			
	return label_map




def remove_boundary(label_map):

	"Remove boundary for those segments which have the label as a result of label reassignment."

	# remove boundary:
	# TODO make this way faster
	


	if label_map.size >0:
	
		t1 = time.time()

		it = np.nditer(label_map, flags=['multi_index'])
		while not it.finished:
			# if boundary_pixel
			if it[0] == 0:
				coords = it.multi_index
				xmin = max(coords[0]-2, coords[0]-1,0)
				xmax = min(coords[0]+2, coords[0]+1, label_map.shape[0])
				ymin = max(coords[1]-2, coords[1]-1,0)
				ymax = min(coords[1]+2, coords[1]+1, label_map.shape[1])
				zmin = max(coords[2]-2, coords[2]-1,0)
				zmax = min(coords[2]+2, coords[2]+1, label_map.shape[2])
				box = label_map[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
				unique_vals = np.unique(box)
					
				if len(unique_vals) == 2:  # 0, and something else
					label_map[coords] = unique_vals[1]
					#print "found 0, replaced with", new_val

			it.iternext()
		t2 = time.time()
		print "time to remove border:", t2-t1

	



#def recolor_label_map_correctly ( label_map, nuclei_collection, seed_collection, segment_collection, seed_segment_collection ):


#	######### COLOR SEED BY NUCLEUS COLOR #########

#	# for each seed segment, color the label map with the index of it parent nucleus.
#	# if the parent nucleus is part of a union, make sure to get the index of the union head.

#	# remove boundary for seeds which will end up having hte same label

#	for i in xrange(len(seed_segment_collection.list_of_seed_segments)-1):
#		seed_segment1 = seed_segment_collection.list_of_seed_segments[i]
#		nucleus_index1 = seed_segment1.seed.nucleus_index

#		for j in xrange(i+1, len(seed_segment_collection.list_of_seed_segments)):

#			seed_segment2 =  seed_segment_collection.list_of_seed_segments[j]
#			nucleus_index2 = seed_segment2.seed.nucleus_index

#			if nucleus_index1 == nucleus_index2:
#				remove_boundary(label_map, seed_segment1, seed_segment2)

##	# remove the boundary for seeds that are neighbor to the nucleus segment they blong to

#	for i in xrange(len(seed_segment_collection.list_of_seed_segments)):
#		seed_segment = seed_segment_collection.list_of_seed_segments[i]
#		parent_nucleus_index = seed_segment.seed.nucleus_index
#		parent_nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[parent_nucleus_index]
#		parent_nucleus = nuclei_collection.nuclei_list[parent_nucleus_list_pos]
#		try:		
#			head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(parent_nucleus)
#		except:
#			pdb.set_trace()
#		segment_label = label_map[head_nucleus.x, head_nucleus.y, head_nucleus.z]
#		try:
#			segment_list_index = segment_collection.segment_label_to_list_index_dict[segment_label]
#		except:
#			pdb.set_trace()
#		segment = segment_collection.list_of_segments[segment_list_index]
#		remove_boundary (label_map, seed_segment, segment)



#	for seed in seed_collection.list_of_seeds:

#		# curent label in the label map (comes from watershed)
#		current_label = label_map[seed.x, seed.y, seed.z]

#		if current_label >1:

#			# get the nucleus index this seed is associated with
#			index_of_parent_nucleus = seed.nucleus_index
#			# get the nucleus object
#			parent_nucleus_list_pos = nuclei_collection.nucleus_index_to_list_pos[index_of_parent_nucleus]
#			parent_nucleus = nuclei_collection.nuclei_list[parent_nucleus_list_pos]
#			# get the head nucleus of the set this nucleus is (in case it was merged with soemthing else)
#			head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(parent_nucleus)
#		
#			# color it by the current color of the head nucleus:
#			new_label = label_map[head_nucleus.x, head_nucleus.y, head_nucleus.z]
#			print current_label ," => ", new_label

#	#		nucleus_list_index = nuclei_collection.nucleus_index_to_list_pos[index_of_parent_nucleus]
#	#		nucleus = nuclei_collection.nuclei_list[nucleus_list_index]
#	#		label_of_segment_at_this_nucleus = label_map[nucleus.x, nucleus.y, nucleus.z]
#	#		new_label = label_of_segment_at_this_nucleus
#		
#			# got the label that this index should have
#			# TODO: dont iterate the whole matrix, only redo the the pixels in that segment
#			label_map = (label_map == current_label) * new_label + (label_map != current_label) * label_map



#	######## COLOR NUCLEUS BY HEAD NUCLEUS COLOR ########

#	# check if we have two segments which have two different head_nuclei
#	# if so, make the minimum bounding box around them, and if it is nonzero, dilate both to color the boundary

#	for i in xrange(len(nuclei_collection.nuclei_list)-1):
#		nucleus1 = nuclei_collection.nuclei_list[i]
#		# for every nucleus, check what nucleus set it belongs to
#		head_nucleus1 = nuclei_collection.get_head_nucleus_in_its_set(nucleus1)
#		segment_label1 = label_map[nucleus1.x, nucleus1.y, nucleus1.z] 

#		for j in xrange(i+1, len(nuclei_collection.nuclei_list)):
#			nucleus2 = nuclei_collection.nuclei_list[j]
#			head_nucleus2 = nuclei_collection.get_head_nucleus_in_its_set(nucleus2)
#			if head_nucleus1.index == head_nucleus2.index:
#				# check if they weren't colored already from past iteration
#				segment_label2 = label_map[nucleus2.x, nucleus2.y, nucleus2.z]		

#				if segment_label1 != segment_label2:
#					# try to get the segment from the segment collection.
#					# if the segment doesnt exist in the collection, it's probably because it was previously merged
#					# with its head nucleus which is a segment in the collection for sure.
#					try:
#						segment_list_index1 = segment_collection.segment_label_to_list_index_dict[segment_label1]
#					except:
#						segment_label1 = label_map[head_nucleus1.x, head_nucleus1.y, head_nucleus1.z] 
#						segment_list_index1 = segment_collection.segment_label_to_list_index_dict[segment_label1]

#					try:
#						segment_list_index2 = segment_collection.segment_label_to_list_index_dict[segment_label2]
#					except:
#						segment_label2 = label_map[head_nucleus2.x, head_nucleus2.y, head_nucleus2.z] 
#						segment_list_index2 = segment_collection.segment_label_to_list_index_dict[segment_label2]

#					segment1 = segment_collection.list_of_segments[segment_list_index1]
#					segment2 = segment_collection.list_of_segments[segment_list_index2]	

#					remove_boundary(label_map, segment1, segment2)



#	# for every nucleus, check if it is part of a union.
#	# if so, recolor that segment with the color of the union head.


#	for nucleus in nuclei_collection.nuclei_list:
#		head_nucleus = nuclei_collection.get_head_nucleus_in_its_set(nucleus)
#		if head_nucleus.index != nucleus.index:
#	
#			current_label = label_map[nucleus.x, nucleus.y, nucleus.z]
#			new_label = label_map[head_nucleus.x, head_nucleus.y, head_nucleus.z]

#			# got the label that this index should have
#			# TODO: dont iterate the whole matrix, only redo the the pixels in that segment
#			label_map = (label_map == current_label) * new_label + (label_map != current_label) * label_map



#		
#	return label_map



#def remove_boundary(label_map, segment1, segment2):

#	# make minimum bounding box around hte border, and dilate each side to cover the boundary

#	xmin = max (segment1.bounding_box.xmin, segment2.bounding_box.xmin)
#	ymin = max (segment1.bounding_box.ymin, segment2.bounding_box.ymin)
#	zmin = max (segment1.bounding_box.zmin, segment2.bounding_box.zmin)

#	xmax = min (segment1.bounding_box.xmax, segment2.bounding_box.xmax)
#	ymax = min (segment1.bounding_box.ymax, segment2.bounding_box.ymax)
#	zmax = min (segment1.bounding_box.zmax, segment2.bounding_box.zmax)


#	# get segment label (either nucleus or seed):
#	try:
#		segment_label1 = segment1.label
#	except:
#		segment_label1 = label_map[segment1.seed.x,segment1.seed.y, segment1.seed.z ]

#	try:
#		segment_label2 = segment2.label	
#	except:
#		segment_label2 = label_map[segment2.seed.x,segment2.seed.y, segment2.seed.z ]

#	cropped_label_map = label_map[xmin:xmax, ymin:ymax, zmin:zmax]
#	mask1 = cropped_label_map == segment_label1
#	mask2 = cropped_label_map == segment_label2
#	mask1 =  ndimage.morphology.binary_dilation(mask1 )
#	mask1 =  ndimage.morphology.binary_dilation(mask1 )
#	mask1 =  ndimage.morphology.binary_dilation(mask1 )
#	mask2 =  ndimage.morphology.binary_dilation(mask2 )
#	mask2 =  ndimage.morphology.binary_dilation(mask2 )
#	mask2 =  ndimage.morphology.binary_dilation(mask2 )
#	mask12 = mask1 * mask2
#	if (mask12.sum() > 0):
#		cropped_label_map = cropped_label_map * (1-mask12) + mask12 * segment_label1
#		label_map[xmin:xmax, ymin:ymax, zmin:zmax] = cropped_label_map
#	


	
