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

	zsum = reduce(lambda a,b: a[2] + b[2], self.list_of_voxel_tuples)		

	box_bounds = segment.bounding_box

	z_mean = int(zsum / len(segment.list_of_voxel_tuples))
	
	
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


def correct_segment_gui (vol, watershed, label, z_default = -1, nuclei_coords = []):

	"""
	GUI to allow the user to interact with the segment to correct.
	Interactions include:
		- add seeds to make new segments.
		- add seeds to modify existing segments.
		- merge existing segments.
	"""


	#print "Nuclei in this cropped section: ", nuclei_coords
	seed_coords = []
	
	fig = pylab.figure(figsize=(12,8))
	fig.canvas.set_window_title("Segment Correction")	
	
	#color_map = pylab.cm.to_rgba(watershed.unique())
	
	print "--------------------------------------------------------------------------------"
	print colored("NO BUTTON TASK SELECTED","blue")
	
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

	button_tasks = set(["MERGE_TWO_LABELS", "ADD_SEEDS_TO_NEW_LABEL", "ADD_SEEDS_TO_EXISTING_LABEL"])


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
			print colored("(Only latest left click counts)", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)
			
		def merge_two_labels(self, event):			
			CellECT.seg_tool.globals.task_index += 1
			CellECT.seg_tool.globals.current_button_task = "MERGE_TWO_LABELS"
			print "--------------------------------------------------------------------------------"
			print colored("MERGE TWO LABELS: Right click for first label, Right click for second label.","blue")
			print colored("(Only latest two right clicks count)", "grey")
			logging.info(CellECT.seg_tool.globals.current_button_task)
						
		def clear_task(self, event):
			CellECT.seg_tool.globals.current_button_task = "NO_TASK_SELECTED"
			print "--------------------------------------------------------------------------------"
			print colored("NO BUTTON TASK SELECTED: You can left/right click anywhere to get info.","blue")
			logging.info(CellECT.seg_tool.globals.current_button_task)
	
	def get_nuclei_at_z(nuclei_coords, z):
	
		nuclei_at_z = filter(lambda x: x[2] == z, nuclei_coords )
		return nuclei_at_z
		
		
	def get_nuclei_at_y(nuclei_coords, y):
	
		nuclei_at_y = filter(lambda x: x[1]== y, nuclei_coords)	
		return nuclei_at_y

	callback = ButtonCallback()
	a_seed_old_label = pylab.axes([0.1, 0.05, 0.22, 0.05])
	a_seed_new_label = pylab.axes([0.34, 0.05, 0.22, 0.05])
	a_merge_two_labels = pylab.axes([0.58, 0.05, 0.18, 0.05])
	a_clear_task = pylab.axes( [0.78, 0.05, 0.12, 0.05 ])
	
	b_seed_old_label = Button(a_seed_old_label, 'Add seeds for an old label')
	b_seed_old_label.on_clicked(callback.seed_old_label)
	b_seed_new_label = Button(a_seed_new_label, "Add 1 seed for a new label")
	b_seed_new_label.on_clicked(callback.seed_new_label)
	b_merge_two_labels = Button(a_merge_two_labels, 'Merge two labels')
	b_merge_two_labels.on_clicked(callback.merge_two_labels)
	b_clear_task = Button(a_clear_task, "Clear task")
	b_clear_task.on_clicked(callback.clear_task)

	fig._seed_for_old_label_button = b_seed_old_label
	fig._seed_for_new_label_button = b_seed_new_label
	fig._merge_two_labels_button = b_merge_two_labels
	fig._clear_task_button = b_clear_task


	if z_default > -1:
		z0 = z_default
	else:		
		z0 = int(np.floor(watershed.shape[2]/2))
		
	y0 = int(watershed.shape[1]/2)
	
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
	max_var_cmap_vol = vol.max()
	l1 =  pylab.imshow(vol[:,:,z0], interpolation="nearest", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol, cmap = "gist_heat")  
	ax1.add_line(line1)  
	ax1.set_aspect(aspect1)
	ax1.axis([0, vol.shape[1], 0, vol.shape[0]])
	ax1.set_autoscale_on(False)
	ax1.invert_yaxis()
	ax1.set_title("x-y slice")

# x-y plane, watershed
	ax2 = pylab.subplot(223)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_ws = watershed.min()
	max_var_cmap_ws = watershed.max()

	l2 =  pylab.imshow(watershed[:,:,z0], interpolation="nearest", cmap = "spectral", vmin = min_var_cmap_ws, vmax = max_var_cmap_ws, picker = True)   #cax = l2
	ax2.add_line(line11) 	
	ax2.set_aspect(aspect1)
	seeds_at_z = get_nuclei_at_z(nuclei_coords, z0)
	if seeds_at_z :
		seeds_at_z = zip(*seeds_at_z)
		ax2.plot(seeds_at_z[0], seeds_at_z[1], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax2.axis([0, vol.shape[1], 0, vol.shape[0]])
	ax2.set_autoscale_on(False)
	ax2.invert_yaxis()
	
# x-z plane, volume
	ax3 = pylab.subplot(222)
	ax3.set_aspect(aspect2)
	pylab.subplots_adjust(bottom=0.25)
	l3 =  pylab.imshow(vol[:,y0,:], interpolation="nearest", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol, cmap = "gist_heat")  
	ax3.add_line(line2) 
	ax3.set_aspect(aspect2)
	ax3.axis([0, vol.shape[2], 0, vol.shape[0]])
	ax3.set_autoscale_on(False)
	ax3.invert_yaxis()
	ax3.set_title("x-z slice")

# x-z plane, watershed
	ax4 = pylab.subplot(224)
	ax4.set_aspect(aspect2)
	pylab.subplots_adjust(bottom=0.25)
	l4 =  pylab.imshow(watershed[:,y0,:], interpolation="nearest", cmap = "spectral", vmin = min_var_cmap_ws, vmax = max_var_cmap_ws, picker = True)   #cax = l2
	ax4.add_line(line22) 
	ax4.set_aspect(aspect2)
	seeds_at_y = get_nuclei_at_y(nuclei_coords, y0)
	if seeds_at_y :
		seeds_at_y = zip(*seeds_at_y)
		ax4.plot(seeds_at_y[0], seeds_at_y[2], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
	ax4.axis([0, vol.shape[2], 0, vol.shape[0]])
	ax4.set_autoscale_on(False)
	ax4.invert_yaxis()
	

	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.15, 0.25, 0.03], axisbg=axcolor)
	s_z = Slider(ax_z, 'z-slice', 0, vol.shape[2]-1, valinit=z0)

	ax_y = pylab.axes([0.6, 0.15, 0.25, 0.03], axisbg=axcolor)
	s_y = Slider(ax_y, 'y-slice', 0, vol.shape[1]-1, valinit=y0)

	def update_z(val):
		z = s_z.val
		# draw lines
		l1.set_data(vol[:,:,z])
		l2.set_data(watershed[:,:,z])
		line2 = Line2D([ z,z], [0,watershed.shape[0]], color = "white", linewidth = 5) 
		del ax3.lines[0]
		ax3.add_line(line2)
		line22 = Line2D([ z,z], [0,watershed.shape[0]], color = "white", linewidth = 5) 
		
		
		try:
			while True:
				del ax4.lines[0]
		except:
			pass
		
		
		# to remove old dots from current view
		
		any_left = True
		while any_left :
			any_left = False
			for i in xrange (len(ax2.lines)):
				if len(ax2.lines[i].get_xydata()) != 2:   # clearly not the vertinal line
					ax2.lines.pop(i)
					any_left = True
					break
				else:  # check if it is the vertical line
					coords = ax2.lines[i].get_xydata()
					if not( coords[0][0] == coords[1][0] and coords[0][1] == 0 and coords[1][1] == watershed.shape[0] ):
						ax2.lines.pop(i)
						any_left = True
						break
						
						
		ax4.add_line(line22)
		
		# draw nuclei
		seeds_at_z = get_nuclei_at_z(nuclei_coords, int(z))
		if seeds_at_z :
			seeds_at_z = zip(*seeds_at_z)
			ax2.plot(seeds_at_z[1], seeds_at_z[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			#print "seeds at z: ", seeds_at_z
		# draw user seeds
		user_seeds_at_z = get_nuclei_at_z(seed_coords, int(z))
		if user_seeds_at_z :
			user_seeds_at_z = zip(*user_seeds_at_z)
			ax2.plot(user_seeds_at_z[1], user_seeds_at_z[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)
		pylab.draw()
		
	def update_y(val):
		y = s_y.val
		# draw lines
		# draw image
		l4.set_data(watershed[:,y,:])
		l3.set_data(vol[:,y,:])
		line1 = Line2D([ y,y], [0,watershed.shape[0]], color = "white", linewidth = 5)  
		l3.set_data(vol[:,y,:])
		# to remove the dots and the vertical line from the other view (cross view)
		try:
			while True:
				del ax2.lines[0]				
		except:
			pass
			

		any_left = True
		while any_left :
			any_left = False
			for i in xrange (len(ax4.lines)):				
				if len(ax4.lines[i].get_xydata()) != 2:   # clearly not the vertinal line
					ax4.lines.pop(i)
					any_left = True
					break
				else:  # check if it is the vertical line
					coords = ax4.lines[i].get_xydata()
					if not( coords[0][0] == coords[1][0] and coords[0][1] == 0 and coords[1][1] == watershed.shape[0] ):
						ax4.lines.pop(i)
						any_left = True
						break

#		
		ax1.add_line(line1)
		line11 = Line2D([ y,y], [0,watershed.shape[0]], color = "white", linewidth = 5) 
		del ax1.lines[0]
		ax2.add_line(line11)
			
		# draw nuclei
		seeds_at_y = get_nuclei_at_y(nuclei_coords, int(y))
		if seeds_at_y :
			seeds_at_y = zip(*seeds_at_y)
			ax4.plot(seeds_at_y[2], seeds_at_y[0], 'w.', markersize = 20., markeredgecolor = "k", markeredgewidth = 3.)
			#print "seeds at y: ", seeds_at_y
		# draw user seeds 
		user_seeds_at_y = get_nuclei_at_y(seed_coords, int(y))
		if user_seeds_at_y :
			user_seeds_at_y = zip(*user_seeds_at_y)
			ax4.plot(user_seeds_at_y[2], user_seeds_at_y[0], 'w*', markersize = 10., markeredgecolor = "k", markeredgewidth = 2.)
			
		pylab.draw()
	
	s_z.on_changed(update_z)
	s_y.on_changed(update_y)
	
	mouse_event = MouseEvent(-2,0,0,0)
	

	list_of_all_mouse_events = []
	
	
	def onpick(event):

		axClicked = event.artist.axes

		
		asc_coordinates = AscidianCoordinates()
		
		# which subplot was clicked:
		if axClicked == ax2:   # the x-y plane
			xval = event.mouseevent.ydata
			yval = event.mouseevent.xdata
			zval = s_z.val
			asc_coordinates.set_coordinates(xval,yval,zval)
		
		elif axClicked == ax4:    # the x-z plane
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
			print "Label %d @ (%d, %d, %d)" % (watershed[int(xval), int(yval), int (zval)], xval, yval, zval)

			
		elif event.mouseevent.button == 1: 
			# left
			
			# draw the current click in one or both plots (if the coords match), unless "clear task" was selected
			
			if CellECT.seg_tool.globals.current_button_task != "NO_TASK_SELECTED":
				
				if int(s_z.val) == zval:
					seed_coords.append ((int(xval), int(yval), int(zval)))
					ax2.plot([yval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					pylab.draw()
				if int(s_y.val) == yval:
					seed_coords.append ((int(xval), int(yval), int(zval)))
					ax4.plot([zval], [xval], 'w*', markersize = 10, markeredgecolor = "k", markeredgewidth = 2.)
					pylab.draw()
			
			print "Seed at: (%d, %d, %d)" % (xval, yval, zval)
		
		


		if event.mouseevent.button in set([1,3]):
			if CellECT.seg_tool.globals.current_button_task in button_tasks:
				list_of_mouse_events_in_ascidian.append( MouseEventInAscidian( asc_coordinates, CellECT.seg_tool.globals.current_button_task, mouse_event, CellECT.seg_tool.globals.task_index ) )
			
	
	
	fig.canvas.mpl_connect('pick_event', onpick)

	
#	for ev in list_of_mouse_events_in_ascidian:
#		print ev.asc_coordinates.xval
#	print '.......'
	

	pylab.show()

	pdb.set_trace()

	return list_of_mouse_events_in_ascidian


