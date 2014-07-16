# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import pylab
from matplotlib.widgets import Slider
import copy
import time
from matplotlib.lines import Line2D
from termcolor import colored
from matplotlib.widgets import Button
import logging
import gc
import matplotlib

# Imports from this project
import CellECT.seg_tool.globals


"""
GUI to allow the user to interact with the segment they want to correct.
Interractions include:
	- adding seeds to create new segments 
	- adding seeds to modify existing segments
	- merging existing segments
"""


def display_segment_to_correct(vol, label_map, segment):

	"""
	Crop the volume and label map around the segment.
	Call the GUI to allow the user to change the segment.
	Return GUI interactions.
	"""


	#x,y,z = zip(*segment.list_of_voxel_tuples)	

	#zsum = reduce(lambda a,b: a[2] + b[2], self.list_of_voxel_tuples)		

	box_bounds = segment.bounding_box

	z_mean = (box_bounds.zmax - box_bounds.zmin)/2 + box_bounds.zmin

	#z_mean = int(zsum / len(segment.list_of_voxel_tuples))
	
	
	pattern = np.zeros( (box_bounds.xmax - box_bounds.xmin, box_bounds.ymax - box_bounds.ymin, box_bounds.zmax - box_bounds.zmin))	
	pattern[::2,::2,:]  = 1
	pattern[1::2, 1::2, :] = 1
	
	cropped_label_map = label_map[ box_bounds.xmin: box_bounds.xmax, box_bounds.ymin: box_bounds.ymax, box_bounds.zmin:box_bounds.zmax] 
	cropped_label_map_copy = copy.deepcopy(cropped_label_map)

	cropped_label_map = np.double(cropped_label_map == segment.label) * pattern * (segment.label+50) +  np.double(cropped_label_map != segment.label) * cropped_label_map

	label_map[ box_bounds.xmin: box_bounds.xmax, box_bounds.ymin: box_bounds.ymax, box_bounds.zmin:box_bounds.zmax] = cropped_label_map

	split_mouse_event, merge_mouse_event1, merge_mouse_event2 = display_volume_two_get_clicks(vol, label_map, z_default = z_mean)

	label_map[ box_bounds.xmin: box_bounds.xmax, box_bounds.ymin: box_bounds.ymax, box_bounds.zmin:box_bounds.zmax] = cropped_label_map_copy
	return split_mouse_event, merge_mouse_event1, merge_mouse_event2


def correct_segment_gui (vol, watershed, label, color_map, vol_max, watershed_max, **kwargs ):

	"""
	GUI to allow the user to interact with the segment to correct.
	Interactions include:
		- add seeds to make new segments.
		- add seeds to modify existing segments.
		- merge existing segments.
	"""

	nuclei_coords = []
	vol_nuclei = None
	z_default = -1
	y_default = -1
	global show_boundary
	show_boundary = False

	if "z_default" in kwargs.keys():
		z_default = int(kwargs["z_default"])

	if "y_default" in kwargs.keys():
		y_default = int(kwargs["y_default"])
	
	if "nuclei_coords" in kwargs.keys():
		nuclei_coords = kwargs["nuclei_coords"]

	if "vol_nuclei" in kwargs.keys():
		vol_nuclei = kwargs["vol_nuclei"]

	#print "Nuclei in this cropped section: ", nuclei_coords
	seed_coords = []
	
	fig = pylab.figure(figsize=(10,10), facecolor='white')
	fig.canvas.set_window_title("Segment Correction")	
	
	#color_map = pylab.cm.to_rgba(watershed.unique())
	
	print "--------------------------------------------------------------------------------"
	print colored("NO BUTTON TASK SELECTED","blue")

	def add_boundary(im_slice, seg_slice):
		mask = np.array(seg_slice == 0)
		in_mask = np.ones_like(mask) - mask
		for i in xrange(3):
			im_slice[:,:,i] = im_slice[:,:,i] * in_mask + 255 * mask


	def get_slice_to_show_z(vol, vol_nuclei,z, seg ):
		z = int(z)
		global show_boundary
		slice_to_show = np.zeros((vol.shape[0], vol.shape[1],3))
		slice_to_show[:,:,0] = vol[:,:,z]
		slice_to_show[:,:,0] = slice_to_show[:,:,0].astype("float")/np.max(slice_to_show[:,:,0])*255
		if not vol_nuclei is None:
			slice_to_show[:,:,1] = vol_nuclei[:,:,z]
			slice_to_show[:,:,1] = slice_to_show[:,:,1].astype("float")/np.max(slice_to_show[:,:,1])*255
		if show_boundary:
			add_boundary(slice_to_show, seg[:,:,z])

		return slice_to_show.astype("uint8")

	def get_slice_to_show_y(vol, vol_nuclei,y, seg ):
		y = int(y)
		global show_boundary
		slice_to_show = np.zeros((vol.shape[0], vol.shape[2],3))
		slice_to_show[:,:,0] = vol[:,y,:]
		slice_to_show[:,:,0] = slice_to_show[:,:,0].astype("float")/np.max(slice_to_show[:,:,0])*255
		if not vol_nuclei is None:
			slice_to_show[:,:,1] = vol_nuclei[:,y,:]
			slice_to_show[:,:,1] = slice_to_show[:,:,1].astype("float")/np.max(slice_to_show[:,:,1])*255
		if show_boundary:
			add_boundary(slice_to_show, seg[:,y,:])
		
		return slice_to_show.astype("uint8")
	
	class MouseEvent:
		def __init__(self, button, xval, yval, axis):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.axis = axis			
			
		def setInfo(self, button, xval, yval,axis):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.axis = axis
	
	class AscidianCoordinates:
		def __init__(self, x = -1,y = -1,z =-1):
			self.xval = int(x)
			self.yval = int(y)
			self.zval = int(z)
		def set_coordinates (self, x,y,z):
			self.xval = int(x)
			self.yval = int(y)
			self.zval = int(z)
			
	class MouseEventInAscidian:	
		def __init__ (self, asc_coordinates, button_task, mouse_event, task_index):		
			self.asc_coordinates = asc_coordinates
			self.right_click = mouse_event.button != 1
			self.button_task = button_task
			self.task_index = task_index
			
	list_of_mouse_events_in_ascidian = []

	button_tasks = set(["MERGE_TWO_LABELS", "ADD_SEEDS_TO_NEW_LABEL", "ADD_SEEDS_TO_EXISTING_LABEL", "ADD_BG_SEED", "DELETE_SEG"])


	CellECT.seg_tool.globals.current_button_task = "NO_TASK_SELECTED"



	class ButtonCallback(object):
	
		def seed_old_label(self,event):
			CellECT.seg_tool.globals.current_button_task = "ADD_SEEDS_TO_EXISTING_LABEL"
			CellECT.seg_tool.globals.task_index += 1
			print "--------------------------------------------------------------------------------"
			print colored("ADD SEEDS FOR EXISTING LABEL: Right click to choose label, Left click to put seeds.","blue")
			#print colored("(Only the latest right click counts. All the left clicks count)", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)
								
		def seed_new_label(self,event):
			CellECT.seg_tool.globals.current_button_task  = "ADD_SEEDS_TO_NEW_LABEL" 
			CellECT.seg_tool.globals.task_index += 1
			print "--------------------------------------------------------------------------------"
			print colored("ADD ONE SEED FOR A NEW LABEL: Left click to put one seed.","blue")
			print colored("(One click per new label)", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)
			
		def merge_two_labels(self, event):			
			CellECT.seg_tool.globals.task_index += 1
			CellECT.seg_tool.globals.current_button_task = "MERGE_TWO_LABELS"
			print "--------------------------------------------------------------------------------"
			print colored("MERGE TWO LABELS: Right click for first label, Right click for second label.","blue")
			print colored("(Only latest two right clicks count)", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)

		def add_bg_seed(self, event):			
			CellECT.seg_tool.globals.task_index += 1
			CellECT.seg_tool.globals.current_button_task = "ADD_BG_SEED"
			print "--------------------------------------------------------------------------------"
			print colored("ADD BACKGROUND SEED: Left click to place background seed.","blue")
			print colored("(One click per seed", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)


		def delete_segment(self, event):
			CellECT.seg_tool.globals.task_index += 1
			CellECT.seg_tool.globals.current_button_task = "DELETE_SEG"
			print "--------------------------------------------------------------------------------"
			print colored("DELETE SEGMENT: Right click to select segment to delete.","blue")
			print colored("(One click per segment", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)
			

		def clear_task(self, event):
			CellECT.seg_tool.globals.current_button_task = "NO_TASK_SELECTED"
			print "--------------------------------------------------------------------------------"
			print colored("NO BUTTON TASK SELECTED: You can left/right click anywhere to get info.","blue")
			logging.info(CellECT.seg_tool.globals.current_button_task)

		def undo_task(self, event):
			print "--------------------------------------------------------------------------------"
			print colored("UNDO LAST TASK: Removing last task click from stack.","blue")
			CellECT.seg_tool.globals.current_button_task = "NO_TASK_SELECTED"
			logging.info("Undo")
			logging.info(CellECT.seg_tool.globals.current_button_task)
			undo_item()
			update_z()
			update_y()
			

		def toggle_boundary(self, event):
			global show_boundary
			show_boundary = not show_boundary
			update_y()
			update_z()
			
		

	
	def get_nuclei_at_z(nuclei_coords, z):
	
		nuclei_at_z = filter(lambda x: x[2] == z, nuclei_coords )
		return nuclei_at_z
		
		
	def get_nuclei_at_y(nuclei_coords, y):
	
		nuclei_at_y = filter(lambda x: x[1]== y, nuclei_coords)	
		return nuclei_at_y


	def undo_item():
		if len(list_of_mouse_events_in_ascidian):
			counter = 1
			item = list_of_mouse_events_in_ascidian[-1]
			# these should really be deques, but since i'm deleting -1 it is still O(1)
			# also, TODO: make sure seed_coords[-1] and list_of_mouse_events_in_ascidian[-1]
			# model the exact same click. 
			
			if not	list_of_mouse_events_in_ascidian[-1].right_click:
				if len(seed_coords):
					del seed_coords[-1]
			if len(list_of_mouse_events_in_ascidian):
				del list_of_mouse_events_in_ascidian[-1]
			task_index = item.task_index

			while len(list_of_mouse_events_in_ascidian) and list_of_mouse_events_in_ascidian[-1].task_index == task_index:
				if len(list_of_mouse_events_in_ascidian):
					if not	list_of_mouse_events_in_ascidian[-1].right_click:
						if len(seed_coords):
							del seed_coords[-1]
					del list_of_mouse_events_in_ascidian[-1]
				counter += 1
		
			print "Removed latest task: %d clicks for %s." % (counter, item.button_task)
		else:
			print "No task to undo"
		print "No task selected."



	def onpick(event):


		axClicked = event.artist.axes

		
		asc_coordinates = AscidianCoordinates()
		
		# which subplot was clicked:
		if axClicked == ax2 or axClicked == ax1:   # the x-y plane
			xval = event.mouseevent.ydata
			yval = event.mouseevent.xdata
			zval = s_z.val
			asc_coordinates.set_coordinates(xval,yval,zval)
		
		elif axClicked == ax4 or axClicked == ax3:   # the x-z plane
			xval = event.mouseevent.ydata
			yval = s_y.val
			zval = event.mouseevent.xdata
			asc_coordinates.set_coordinates(xval,yval,zval)
			
		
		xval = asc_coordinates.xval
		yval = asc_coordinates.yval
		zval = asc_coordinates.zval
				
		mouse_event = MouseEvent(event.mouseevent.button, event.mouseevent.xdata, event.mouseevent.ydata, axClicked)	
		

		if event.mouseevent.button == 3:
			# right		
			label = watershed[int(xval), int(yval), int (zval)]
			text = "Label %d" % label
			if label == 1:
				text = "Background"
			if label == 0:
				text = "Border"
			print "%s @ (%d, %d, %d)" % (text, xval, yval, zval)

			
		elif event.mouseevent.button == 1: 
			# left
			
			# draw the current click in one or both plots (if the coords match), unless "clear task", "merge 2" or "delete" was selected

			
			if not CellECT.seg_tool.globals.current_button_task in set(["NO_TASK_SELECTED", "MERGE_TWO_LABELS", "DELETE_SEG"]):

				# if second seed for new label click in a row, then make a new label task:
				if CellECT.seg_tool.globals.current_button_task == "ADD_SEEDS_TO_NEW_LABEL":
					if len(list_of_mouse_events_in_ascidian) and list_of_mouse_events_in_ascidian[-1].button_task == "ADD_SEEDS_TO_NEW_LABEL":
						callback.seed_new_label(None)


				if int(s_z.val) == zval:
					seed_coords.append ((int(xval), int(yval), int(zval)))
					ax2.plot([yval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					ax1.plot([yval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					
					pylab.draw()
				if int(s_y.val) == yval:
					# don't add it twice in case it was added in the above if-statement
					# this happens when you click on the intersection of the two planes
					if len(seed_coords)>0:
						if seed_coords[-1] != (int(xval), int(yval), int(zval)):
							seed_coords.append ((int(xval), int(yval), int(zval)))
					else: 
						seed_coords.append ((int(xval), int(yval), int(zval)))
					ax4.plot([zval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					ax3.plot([zval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					pylab.draw()
			
				print "Seed at: (%d, %d, %d)" % (xval, yval, zval)
			else:
				print "Clicked at: (%d, %d, %d)" % (xval, yval, zval)
		
		


		if event.mouseevent.button in set([1,3]):
			if CellECT.seg_tool.globals.current_button_task in button_tasks:
				list_of_mouse_events_in_ascidian.append( MouseEventInAscidian( asc_coordinates, CellECT.seg_tool.globals.current_button_task, mouse_event, CellECT.seg_tool.globals.task_index ) )
			




	def update_points():

		# delete all points/lines from all views:

		try:
			while True:
				ax1.lines.pop()
		except:
			pass

		try:
			while True:
				ax2.lines.pop()
		except:
			pass

		try:
			while True:
				ax3.lines.pop()
		except:
			pass

		try:
			while True:
				ax4.lines.pop()
		except:
			pass


		y = s_y.val
		z = s_z.val

		# draw vertical lines in all views:

		line1 = Line2D([ y,y], [0,watershed.shape[0]], color = "white", linewidth = 5)
		ax1.add_line(line1)
		line11 = Line2D([ y,y], [0,watershed.shape[0]], color = "white", linewidth = 5)
		ax2.add_line(line11)

		line2 = Line2D([ z,z], [0,watershed.shape[0]], color = "white", linewidth = 5)
		ax3.add_line(line2)
		line22 = Line2D([ z,z], [0,watershed.shape[0]], color = "white", linewidth = 5)
		ax4.add_line(line22)
		
		# draw nuclei in y-planes:
		seeds_at_y = get_nuclei_at_y(nuclei_coords, int(y))
		if seeds_at_y :
			seeds_at_y = zip(*seeds_at_y)
			ax4.plot(seeds_at_y[2], seeds_at_y[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			ax3.plot(seeds_at_y[2], seeds_at_y[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			#print "seeds at y: ", seeds_at_y

		# draw user seeds in y-planes 
		user_seeds_at_y = get_nuclei_at_y(seed_coords, int(y))
		if user_seeds_at_y :
			user_seeds_at_y = zip(*user_seeds_at_y)
			ax4.plot(user_seeds_at_y[2], user_seeds_at_y[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)
			ax3.plot(user_seeds_at_y[2], user_seeds_at_y[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)

		# draw nuclei in z planes
		seeds_at_z = get_nuclei_at_z(nuclei_coords, int(z))
		if seeds_at_z :
			seeds_at_z = zip(*seeds_at_z)
			ax2.plot(seeds_at_z[1], seeds_at_z[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			ax1.plot(seeds_at_z[1], seeds_at_z[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			#print "seeds at z: ", seeds_at_z

		# draw user seeds in z planes
		user_seeds_at_z = get_nuclei_at_z(seed_coords, int(z))
		if user_seeds_at_z :
			user_seeds_at_z = zip(*user_seeds_at_z)
			ax2.plot(user_seeds_at_z[1], user_seeds_at_z[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)
			ax1.plot(user_seeds_at_z[1], user_seeds_at_z[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)


		pylab.draw()



	def update_z(val=None):

		z = int(s_z.val)
		# draw lines
		l1.set_data(get_slice_to_show_z(vol, vol_nuclei,z,watershed))
		l2.set_data(watershed[:,:,z])
		update_points()

		
	def update_y(val= None):

		y = int(s_y.val)
		# draw lines
		# draw image
		l4.set_data(watershed[:,y,:])
		l3.set_data(get_slice_to_show_y(vol, vol_nuclei,y, watershed))
		update_points()
		



	def next_z_click(event):

		z = int(s_z.val)
		z += 1

		if z<vol.shape[2]:
			s_z.set_val(z)
			update_z()	


	def prev_z_click(event):

		z = int(s_z.val)
		z -= 1

		if z>=0:
			s_z.set_val( z)
			update_z()	


	def next_y_click(event):

		y = int(s_y.val)
		y += 1

		if y<vol.shape[1]:
			s_y.set_val( y)
			update_y()		


	def prev_y_click(event):

		y = int(s_y.val)
		y -= 1

		if y>=0:
			s_y.set_val(y)
			update_y()





	callback = ButtonCallback()
	a_seed_old_label = pylab.axes([0.07, 0.06, 0.16, 0.04])
	a_seed_new_label = pylab.axes([0.24, 0.06, 0.13, 0.04])
	a_merge_two_labels = pylab.axes([0.38, 0.06, 0.17, 0.04])
	a_delete_seg = pylab.axes([0.56, 0.06, 0.16, 0.04])
	a_bg_seed = pylab.axes([0.73, 0.06, 0.20, 0.04])

	
	a_clear_task = pylab.axes( [0.35, 0.01, 0.09, 0.04 ])
	a_undo = pylab.axes( [0.45, 0.01, 0.13, 0.04 ])
	a_toggle = pylab.axes([0.59, 0.01, 0.13, 0.04])


	a_next_z = pylab.axes([0.34, 0.125, 0.025, 0.025])
	a_prev_z = pylab.axes([0.30, 0.125, 0.025, 0.025])
	a_next_y = pylab.axes([0.74, 0.125, 0.025, 0.025])
	a_prev_y = pylab.axes([0.70, 0.125, 0.025, 0.025])
	
	b_seed_old_label = Button(a_seed_old_label, 'Modify segment')
	b_seed_old_label.on_clicked(callback.seed_old_label)
	b_seed_new_label = Button(a_seed_new_label, "New segment")
	b_seed_new_label.on_clicked(callback.seed_new_label)
	b_merge_two_labels = Button(a_merge_two_labels, 'Merge 2 segments')
	b_merge_two_labels.on_clicked(callback.merge_two_labels)
	b_clear_task = Button(a_clear_task, "No task")
	b_clear_task.on_clicked(callback.clear_task)
	b_undo_task = Button(a_undo, "Undo task")
	b_undo_task.on_clicked(callback.undo_task)
	b_toggle = Button(a_toggle, "Show Borders")
	b_toggle.on_clicked(callback.toggle_boundary)
	b_bg_seed = Button(a_bg_seed, "Background segment")
	b_bg_seed.on_clicked(callback.add_bg_seed)
	b_delete_seg = Button(a_delete_seg, "Delete segment")
	b_delete_seg.on_clicked(callback.delete_segment)


	b_next_z = Button(a_next_z, ">")
	b_next_z.on_clicked(next_z_click)
	b_prev_z = Button(a_prev_z, "<")
	b_prev_z.on_clicked(prev_z_click)
	b_next_y = Button(a_next_y, ">")
	b_next_y.on_clicked(next_y_click)
	b_prev_y = Button(a_prev_y, "<")
	b_prev_y.on_clicked(prev_y_click)

	fig._seed_for_old_label_button = b_seed_old_label
	fig._seed_for_new_label_button = b_seed_new_label
	fig._merge_two_labels_button = b_merge_two_labels
	fig._clear_task_button = b_clear_task
	fig._undo_button = b_undo_task
	fig._toggle = b_toggle
	fig._bg_seed = b_bg_seed
	fig._b_next_z = b_next_z
	fig._b_prev_z = b_prev_z
	fig._b_next_y = b_next_y
	fig._b_prev_y = b_prev_y
	fig._b_delete_seg = b_delete_seg


	if z_default > -1:
		z0 = z_default
	else:		
		z0 = int(np.floor(watershed.shape[2]/2))

	if y_default > -1:
		y0 = y_default
	else:		
		y0 = int(np.floor(watershed.shape[2]/2))
		
	#y0 = int(watershed.shape[1]/2)
	
	aspect1 = abs(watershed.shape[1]/float(watershed.shape[0]))
	aspect2 = abs(watershed.shape[2]/float(watershed.shape[0]))
	
	line1 = Line2D([ y0,y0], [0,watershed.shape[0]], color = "white", linewidth = 5)
	line11 = Line2D([ y0,y0], [0,watershed.shape[0]], color = "white", linewidth = 5)                                    
	line2 = Line2D([ z0,z0], [0,watershed.shape[0]], color = "white", linewidth = 5)                                     
	line22 = Line2D([ z0,z0], [0,watershed.shape[0]], color = "white", linewidth = 5)
	 
	z0 = int(z0) 


# x-y plane, volume
	ax1 = pylab.subplot(221)	
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_vol = vol.min()
	max_var_cmap_vol = vol_max
	l1 =  pylab.imshow(get_slice_to_show_z(vol, vol_nuclei, z0, watershed), interpolation="nearest", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol, cmap = "gist_heat", picker = True)  
	ax1.add_line(line1)  
	ax1.set_aspect(aspect1)
	seeds_at_z = get_nuclei_at_z(nuclei_coords, z0)
	if seeds_at_z :
		seeds_at_z = zip(*seeds_at_z)
		ax1.plot(seeds_at_z[1], seeds_at_z[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax1.axis([0, vol.shape[1], 0, vol.shape[0]])
	ax1.set_autoscale_on(False)
	ax1.invert_yaxis()
	ax1.set_title("x-y slice")

# x-y plane, watershed
	ax2 = pylab.subplot(223)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_ws = 0
	max_var_cmap_ws = watershed_max

	l2 =  pylab.imshow(watershed[:,:,z0], interpolation="nearest", cmap = color_map, vmin = min_var_cmap_ws, vmax = max_var_cmap_ws, picker = True)   #cax = l2
	ax2.add_line(line11) 	
	ax2.set_aspect(aspect1)
	seeds_at_z = get_nuclei_at_z(nuclei_coords, z0)
	if seeds_at_z :
		seeds_at_z = zip(*seeds_at_z)
		ax2.plot(seeds_at_z[1], seeds_at_z[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax2.axis([0, vol.shape[1], 0, vol.shape[0]])
	ax2.set_autoscale_on(False)
	ax2.invert_yaxis()
	
# x-z plane, volume
	ax3 = pylab.subplot(222)
	ax3.set_aspect(aspect2)
	pylab.subplots_adjust(bottom=0.25)
	l3 =  pylab.imshow(get_slice_to_show_y(vol, vol_nuclei,y0, watershed), interpolation="nearest", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol, cmap = "gist_heat", picker = True)  
	ax3.add_line(line2) 
	ax3.set_aspect(aspect2)
	seeds_at_y = get_nuclei_at_y(nuclei_coords, y0)
	if seeds_at_y :
		seeds_at_y = zip(*seeds_at_y)
		ax3.plot(seeds_at_y[2], seeds_at_y[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax3.axis([0, vol.shape[2], 0, vol.shape[0]])
	ax3.set_autoscale_on(False)
	ax3.invert_yaxis()
	ax3.set_title("x-z slice")

# x-z plane, watershed
	ax4 = pylab.subplot(224)
	ax4.set_aspect(aspect2)
	pylab.subplots_adjust(bottom=0.25)
	l4 =  pylab.imshow(watershed[:,y0,:], interpolation="nearest", cmap = color_map, vmin = min_var_cmap_ws, vmax = max_var_cmap_ws, picker = True)   #cax = l2
	ax4.add_line(line22) 
	ax4.set_aspect(aspect2)
	seeds_at_y = get_nuclei_at_y(nuclei_coords, y0)
	if seeds_at_y :
		seeds_at_y = zip(*seeds_at_y)
		ax4.plot(seeds_at_y[2], seeds_at_y[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax4.axis([0, vol.shape[2], 0, vol.shape[0]])
	ax4.set_autoscale_on(False)
	ax4.invert_yaxis()

	

	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.16, 0.25, 0.03], axisbg=axcolor)
	s_z = Slider(ax_z, 'z-slice', 0, vol.shape[2]-1, valinit=z0)

	ax_y = pylab.axes([0.6, 0.16, 0.25, 0.03], axisbg=axcolor)
	s_y = Slider(ax_y, 'y-slice', 0, vol.shape[1]-1, valinit=y0)

	s_z.on_changed(update_z)
	s_y.on_changed(update_y)


	fig.canvas.mpl_connect('pick_event', onpick)
	pylab.show()

	
	mouse_event = MouseEvent(-2,0,0,0)
	

	list_of_all_mouse_events = []
	
	

	
	

	

	
#	for ev in list_of_mouse_events_in_ascidian:
#		print ev.asc_coordinates.xval
#	print '.......'
	

	if matplotlib.get_backend().lower() != "GTK3Agg".lower() and matplotlib.get_backend().lower() != "GTKAgg".lower():
		pylab.close()
		gc.collect()

	return list_of_mouse_events_in_ascidian, fig


