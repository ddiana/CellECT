# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import numpy as np
import pdb
import pylab
from matplotlib.widgets import Slider
import time
from matplotlib.lines import Line2D
from termcolor import colored
from matplotlib.widgets import Button
import os
import logging

# Imports from this project
from CellECT.seg_tool.gui import correct_segment_gui as seg_gui
from CellECT.seg_tool.seg_io import save_all

import CellECT.seg_tool.globals


"""
Compute uncertainty map which shows the cellness metric for each segment.
Color code segments based on the certainty of the segmentation.
Display the uncertainty map and the segmentation, allow the user to select which
segment they want to correct.
Also possible to save current result or load an old result.
"""


def get_segment_uncertainty_map(watershed, collection_of_segments, classified_segments):
	"""
	Make segment uncertainty map.
	"""

	# classified_segments = [ (label, class_prediction, discriminant_value ) ]

	uncertainty_map = np.zeros(watershed.shape)
	
	for i in xrange(len(classified_segments[0])):
	
		for voxel in collection_of_segments.list_of_segments[i].list_of_voxel_tuples:
			uncertainty_map[voxel] = classified_segments[2][i]


	minval = uncertainty_map.min()
	uncertainty_map -= minval
	uncertainty_map = (uncertainty_map != abs(minval)) * (uncertainty_map.max() - uncertainty_map)
	
#	uncertainty_map = (uncertainty_map == 0) * np.min(uncertainty_map) + (uncertainty_map !=0) * uncertainty_map
#	uncertainty_map += np.min(uncertainty_map)
#	uncertainty_map /= float(np.max(uncertainty_map))
#	uncertainty_map *= 255
	
	

	return uncertainty_map



def show_uncertainty_map_and_get_feedback(vol, watershed, segment_collection, classified_segments, nuclei_collection, seed_collection, seed_segment_collection, watershed_old, z_default = -1	):

	"""
	Main GUI which shows uncertainty map and allows user to select which segment
	they want to modify.
	Also possible to load an old result or save the current result.	
	"""

	logging.info ("STARTING USER FEEDBACK")
	print colored("============================== START USER FEEDBACK =============================", "yellow")

	nuclei_coords = [ (nucleus.x, nucleus.y, nucleus.z) for nucleus in nuclei_collection.nuclei_list ]

	uncertainty_map = get_segment_uncertainty_map(watershed, segment_collection, classified_segments)
	


	CellECT.seg_tool.globals.task_index = 0
	
	fig = pylab.figure(figsize=(15,8))
	fig.canvas.set_window_title("Cell Confidence Map")
	
	list_of_all_mouse_events = []
	class MouseEvent:
		def __init__(self, button, xval, yval, zval):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.zval = zval
			
		def setInfo(self, button, xval, yval,zval):
			self.button = button # left or right
			self.xval = xval
			self.yval = yval
			self.zval = zval
	
	class MouseEventsFromSegmentGUI(object):
	
		def __init__ (self, bounding_box, list_of_mouse_events_in_cropped_ascidian):
		
			self.bounding_box = bounding_box
			self.list_of_cropped_ascidian_events = list_of_mouse_events_in_cropped_ascidian


	if z_default > -1:
		z0 = z_default
	else:
		z0 = int(np.floor(watershed.shape[2]/2))

	ax1 = pylab.subplot(141)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_vol = vol.min()
	max_var_cmap_vol = vol.max()
	l1 =  pylab.imshow(vol[:,:,z0], interpolation="nearest", cmap = "gist_heat", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol)  
	pylab.axis()#[0, vol1.shape[0], 0, vol1.shape[1]])
	ax1.set_title("Input Volume")
	

	ax2 = pylab.subplot(142)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_uncert = uncertainty_map.min()
	max_var_cmap_uncert = uncertainty_map.max()
	l2 =  pylab.imshow(uncertainty_map[:,:,z0], interpolation="nearest", cmap = "PRGn", vmin= min_var_cmap_uncert, vmax = max_var_cmap_uncert, picker = True)   #cax = l2
	pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
	ax2.set_title("Uncertainty Map")

	ax3 = pylab.subplot(143)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_uncert = watershed.min()
	max_var_cmap_uncert = watershed.max()
	l3 =  pylab.imshow(watershed[:,:,z0], interpolation="nearest", cmap = "spectral", vmin= min_var_cmap_uncert, vmax = max_var_cmap_uncert, picker = True)   #cax = l2
	pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
	ax3.set_title("Segmentation Label Map")

	dif_watershed = (watershed==0).astype("int32") - (watershed_old==0).astype("int32")
	ax4 = pylab.subplot(144)
	pylab.subplots_adjust(bottom=0.25)
	min_var_cmap_uncert = dif_watershed .min()
	max_var_cmap_uncert = dif_watershed .max()
	l4 =  pylab.imshow(dif_watershed [:,:,z0], interpolation="nearest", cmap = "RdYlGn", vmin= -1, vmax = 1)   #cax = l2
	pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
	ax4.set_title("Difference From Previous")


	axcolor = 'lightgoldenrodyellow'
	ax_z = pylab.axes([0.2, 0.15, 0.65, 0.03], axisbg=axcolor)

	s_z = Slider(ax_z, 'z-slice', 0, vol.shape[2]-1, valinit=z0)


	def load_last_save_callback(event):
	
		can_load = False
		file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "nuclei.xml"
		if 	os.path.exists(file_name):
			file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "seeds.xml"
			if os.path.exists(file_name):
				file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "label_map.mat"
				if os.path.exists(file_name):
					can_load = True

		CellECT.seg_tool.globals.should_load_last_save

		if can_load:
			CellECT.seg_tool.globals.should_load_last_save = True
		else:
			print "Cannot load previous state. Files missing."
			CellECT.seg_tool.globals.should_load_last_save = False

		pylab.close()

	def save_current_status_callback(event):
		save_all.save_current_status(nuclei_collection, seed_collection, segment_collection, seed_segment_collection, watershed)

	a_load = pylab.axes([0.2, 0.05, 0.3, 0.05])
	a_save = pylab.axes([0.55, 0.05, 0.3, 0.05])
	
	b_load = Button(a_load, 'Load last save (if any)')
	b_load.on_clicked(load_last_save_callback)
	b_save = Button(a_save, "Save current state")
	b_save.on_clicked(save_current_status_callback)





	def update(val):
		z = s_z.val
		l1.set_data(vol[:,:,z])
		l2.set_data(uncertainty_map[:,:,z])
		l3.set_data(watershed[:,:,z])
		l4.set_data(dif_watershed[:,:,z])
		pylab.draw()
	
	s_z.on_changed(update)
	
	mouse_event = MouseEvent(-2,0,0,0)
	
	
	def onpick(event):


		xval = event.mouseevent.xdata
		yval = event.mouseevent.ydata
		zval = s_z.val

		label = watershed[int(yval), int(xval), int(zval)]

		if label < 2:
			print "Border/Backdround selected. Try again."
		
		else:
			if event.mouseevent.button != 1:
				# right click to mark as correct
				# print "Correct segment: ", label, "@", int(yval), int(xval), int(zval)
			
				# TODO: KEEP CORRECT LABEL INFORMATION

				pass
			
			else:
				message = "Opening GUI for segment: Label %d @ (%d, %d, %d)" % (label, int(yval), int(xval), int(zval))
				print message
				logging.info(message)
			
			
				segment_index = segment_collection.segment_label_to_list_index_dict[label]
			
				bounding_box = segment_collection.list_of_segments[segment_index].bounding_box
				bounding_box.extend_by(10,vol.shape)
			
				cropped_nuclei_coords = filter(lambda nucl: nucl[0] > bounding_box.xmin and nucl[0] < bounding_box.xmax and nucl[1] > bounding_box.ymin and nucl[1] < bounding_box.ymax and nucl[2] > bounding_box.zmin and nucl[2] < bounding_box.zmax, nuclei_coords )
				cropped_nuclei_coords = [ (item[0] - bounding_box.xmin, item[1] - bounding_box.ymin, item[2] - bounding_box.zmin ) for item in cropped_nuclei_coords]
			
				cropped_vol = vol[bounding_box.xmin : bounding_box.xmax, bounding_box.ymin: bounding_box.ymax, bounding_box.zmin: bounding_box.zmax]
				cropped_watershed = watershed[bounding_box.xmin : bounding_box.xmax, bounding_box.ymin: bounding_box.ymax, bounding_box.zmin: bounding_box.zmax]

				list_of_mouse_events_in_cropped_ascidian = seg_gui.correct_segment_gui (cropped_vol, cropped_watershed, label, z_default = zval - bounding_box.zmin, nuclei_coords =  cropped_nuclei_coords)
					

				list_of_all_mouse_events.append( MouseEventsFromSegmentGUI(bounding_box, list_of_mouse_events_in_cropped_ascidian ))

			
	
	fig.canvas.mpl_connect('pick_event', onpick)

	pylab.show()
	pylab.close()
	
	print colored("=============================== END USER FEEDBACK ==============================","yellow")
	logging.info("ENDING USER FEEDBACK")

	return list_of_all_mouse_events



