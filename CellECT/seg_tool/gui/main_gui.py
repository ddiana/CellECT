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
import random
import matplotlib


# Imports from this project
from CellECT.seg_tool.gui import correct_segment_gui as seg_gui
from CellECT.seg_tool.seg_io import save_all
from CellECT.seg_tool.cellness_metric import merge_predictor
from CellECT.seg_tool.seg_utils import bounding_box as bbx_module

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




		




def show_uncertainty_map_and_get_feedback(vol, watershed, segment_collection, classified_segments, nuclei_collection, seed_collection, seed_segment_collection, watershed_old, correct_labels, bg_seeds , **kwargs):

	"""
	Main GUI which shows uncertainty map and allows user to select which segment
	they want to modify.
	Also possible to load an old result or save the current result.	
	"""

	z_default = -1
	vol_nuclei = None
	


	if "z_default" in kwargs.keys():
		z_default = kwargs["z_default"]

	if "vol_nuclei" in kwargs.keys():
		vol_nuclei = kwargs["vol_nuclei"]

	logging.info ("STARTING USER FEEDBACK")
	print colored("============================== START USER FEEDBACK =============================", "yellow")

	head_nuclei =  filter(lambda nucleus: nucleus == nuclei_collection.get_head_nucleus_in_its_set(nucleus), nuclei_collection.nuclei_list )

	nuclei_coords = [ (nucleus.x, nucleus.y, nucleus.z) for nucleus in head_nuclei ]

	uncertainty_map = get_segment_uncertainty_map(watershed, segment_collection, classified_segments)
	
	sub_figs = []

	CellECT.seg_tool.globals.task_index = 0
	
	fig = pylab.figure(figsize=(15,8), facecolor='white')
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


	watershed_max = watershed.max()

	colors = [(0,0,0)] + [(0,0,0)] + [(random.random(),random.random(),random.random()) for i in xrange(watershed_max)]

	color_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=watershed_max)

	global merge_pred
	merge_pred = merge_predictor.MergePredictor(segment_collection, vol, vol_nuclei, watershed, color_map)

	

	ax1 = pylab.subplot(141)
	pylab.subplots_adjust(bottom=0.25)

	l1 = None
	min_var_cmap_vol = vol.min()
	max_var_cmap_vol = vol.max()
	# if nuclear channel, show 3 color image. else show just membrane
	slice_to_show = np.zeros((vol.shape[0], vol.shape[1], 3))
	slice_to_show[:,:,0] = vol[:,:,z0]
	slice_to_show[:,:,0] = slice_to_show[:,:,0].astype("float")/np.max(slice_to_show[:,:,0])*255

	if not vol_nuclei is None:
		slice_to_show[:,:,1] = vol_nuclei[:,:,z0]		
		slice_to_show[:,:,1] = slice_to_show[:,:,1].astype("float")/np.max(slice_to_show[:,:,1])*255

	l1 =  pylab.imshow(slice_to_show.astype("uint8"), interpolation="nearest")  

	
	#l1 =  pylab.imshow(vol[:,:,z0], interpolation="nearest", cmap = "gist_heat", vmin = min_var_cmap_vol, vmax = max_var_cmap_vol)  

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
	min_var_cmap_ws = 0
	max_var_cmap_ws = watershed.max()
	l3 =  pylab.imshow(watershed[:,:,z0], interpolation="nearest", cmap = color_map, vmin= min_var_cmap_ws, vmax = max_var_cmap_ws, picker = True)   #cax = l2
	pylab.axis()#[0, vol2.shape[0], 0, vol2.shape[1]])
	ax3.set_title("Segmentation Label Map")

	dif_watershed = (watershed==0).astype("int32") - (watershed_old==0).astype("int32")
	ax4 = pylab.subplot(144)
	pylab.subplots_adjust(bottom=0.25)
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
		save_all.save_current_status(nuclei_collection, seed_collection, segment_collection, seed_segment_collection, watershed, bg_seeds)

	def add_boundary(im_slice, seg_slice):
		mask = np.array(seg_slice == 0)
		in_mask = np.ones_like(mask) - mask
		for i in xrange(3):
			im_slice[:,:,i] = im_slice[:,:,i] * in_mask + 255 * mask


	def predict_correction(event):

		global merge_pred
		merge_pred.next_merge()


	global show_border_toggle
	show_border_toggle = False

	def show_border(event):

		global show_border_toggle
		show_border_toggle = not show_border_toggle
		update()

	def rerun(event):
		
		for subfig in sub_figs:
			pylab.close(subfig)
		pylab.close(fig)

	a_border = pylab.axes([0.10, 0.05, 0.08, 0.05])
	a_load = pylab.axes([0.20, 0.05, 0.16, 0.05])
	a_save = pylab.axes([0.38, 0.05, 0.16, 0.05])
	a_rerun = pylab.axes([0.56, 0.05, 0.22, 0.05])
	a_predict = pylab.axes([0.80, 0.05, 0.15, 0.05])
	
	
	b_load = Button(a_load, 'Load last save (if any)')
	b_load.on_clicked(load_last_save_callback)
	b_save = Button(a_save, "Save current state")
	b_save.on_clicked(save_current_status_callback)
	b_rerun = Button(a_rerun, "RERUN with user feedback")
	b_rerun.on_clicked(rerun)
	b_border = Button(a_border, "Border")
	b_border.on_clicked(show_border)
	b_predict = Button(a_predict, "Suggest me!")
	b_predict.on_clicked(predict_correction)




	


	def update(val = None):
		z = s_z.val

		slice_to_show = np.zeros((vol.shape[0], vol.shape[1], 3))
		slice_to_show[:,:,0] = vol[:,:,z]
		slice_to_show[:,:,0] = slice_to_show[:,:,0].astype("float")/np.max(slice_to_show[:,:,0])*255
		if not vol_nuclei is None:
			slice_to_show[:,:,1] = vol_nuclei[:,:,z]
			slice_to_show[:,:,1] = slice_to_show[:,:,1].astype("float")/np.max(slice_to_show[:,:,1])*255

		if show_border_toggle:
			add_boundary(slice_to_show, watershed[:,:,z])

		l1.set_data(slice_to_show.astype("uint8"))
		l2.set_data(uncertainty_map[:,:,z])
		l3.set_data(watershed[:,:,z])
		l4.set_data(dif_watershed[:,:,z])
		pylab.draw()
	
	s_z.on_changed(update)
	
	mouse_event = MouseEvent(-2,0,0,0)

	global zoom_x_range_cur
	zoom_x_range_cur = watershed.shape[0]
	global zoom_y_range_cur
	zoom_y_range_cur = watershed.shape[1]
	zoom_x_range_max = watershed.shape[0]
	zoom_y_range_max = watershed.shape[1]
	zoom_x_range_min = zoom_x_range_max / 10
	zoom_y_range_min = zoom_y_range_max / 10
		
	def scroll_zoom(event):

		return

		base_scale = 1.5
		global zoom_x_range_cur
		global zoom_y_range_cur

		cur_xlim = ax1.get_xlim()
		cur_ylim = ax1.get_ylim()
#		cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
#		cur_yrange = (cur_ylim[0] - cur_ylim[1])*.5
		xdata = event.xdata # get event x location
		ydata = event.ydata # get event y location
		if event.button == 'up':
			# deal with zoom in
			scale_factor = 1/base_scale
		elif event.button == 'down':
			# deal with zoom out
			scale_factor = base_scale
		else:
			# deal with something that should never happen
			scale_factor = 1
			print event.button
		# set new limits


			xrange_temp = zoom_x_range_cur * scale_factor
			yrange_temp = zoom_y_range_cur * scale_factor

			if xrange_temp < zoom_x_range_max and \
               xrange_temp > zoom_x_range_min and \
               yrange_temp < zoom_y_range_max and \
               yrange_temp > zoom_y_range_min:

				zoom_x_range_cur = xrange_temp
				zoom_y_range_cur = yrange_temp

			else:
				return

			image_x = watershed.shape[0]
			image_y = watershed.shape[1]

			xmin = xdata - zoom_x_range_cur/2
			xmax = xdata + zoom_x_range_cur/2
			if xmin <0:
				xmin = 0
				xmax = zoom_x_range_cur*2

			if xmax > image_x:
				xmax = image_x
				xmin = image_x - zoom_x_range_cur*2
			
					
			ymin = ydata - zoom_y_range_cur/2
			ymax = ydata + zoom_y_range_cur/2
			if ymin <0:
				ymin = 0
				ymax = zoom_y_range_cur*2

			if ymax > image_y:
				ymax = image_y
				ymin = image_y - zoom_y_range_cur*2


			for ax in [ax1, ax2, ax3, ax4]:

				ax.set_xlim([xmin, xmax])
				ax.set_ylim([ymin, ymax])

		pylab.draw() # force re-draw

	def onpick(event):

		xval = event.mouseevent.xdata
		yval = event.mouseevent.ydata
		zval = s_z.val

		label = watershed[int(yval), int(xval), int(zval)]

		if label < 1:
			print "Border selected. Try again."
	
		else:

			if label ==1:
				print "Background selected. Showing whole volume."
	
			if event.mouseevent.button == 3 and label>1:

				# mark correct segments with right click

				message = "Marking segment with Label %d @ (%d, %d, %d) as correct." % (label, int(yval), int(xval), int(zval))
				print message

				
				correct_labels.add(label)
			
			elif event.mouseevent.button ==1:
				message = "Opening GUI for segment: Label %d @ (%d, %d, %d)" % (label, int(yval), int(xval), int(zval))
				print message
			
				logging.info(message)

				bounding_box = bbx_module.BoundingBox(0, vol.shape[0], 0, vol.shape[1], 0, vol.shape[2])
			
				if label > 1: # not background

					segment_index = segment_collection.segment_label_to_list_index_dict[label]
			
					bounding_box = segment_collection.list_of_segments[segment_index].bounding_box
					bounding_box.extend_by(10,vol.shape)
			
				cropped_nuclei_coords = filter(lambda nucl: nucl[0] > bounding_box.xmin and nucl[0] < bounding_box.xmax and nucl[1] > bounding_box.ymin and nucl[1] < bounding_box.ymax and nucl[2] > bounding_box.zmin and nucl[2] < bounding_box.zmax, nuclei_coords )
				cropped_nuclei_coords = [ (item[0] - bounding_box.xmin, item[1] - bounding_box.ymin, item[2] - bounding_box.zmin ) for item in cropped_nuclei_coords]			


				cropped_vol = vol[bounding_box.xmin : bounding_box.xmax, bounding_box.ymin: bounding_box.ymax, bounding_box.zmin: bounding_box.zmax]
				cropped_watershed = watershed[bounding_box.xmin : bounding_box.xmax, bounding_box.ymin: bounding_box.ymax, bounding_box.zmin: bounding_box.zmax]

				cropped_vol_nuclei = None
				if not vol_nuclei is None:
					cropped_vol_nuclei = vol_nuclei[bounding_box.xmin : bounding_box.xmax, bounding_box.ymin: bounding_box.ymax, bounding_box.zmin: bounding_box.zmax]

				#print "box: %d, %d" % (bounding_box.xmin, bounding_box.ymin)
				list_of_mouse_events_in_cropped_ascidian, temp_fig = seg_gui.correct_segment_gui (cropped_vol, cropped_watershed, label, color_map, vol.max(), watershed_max,  z_default = zval - bounding_box.zmin,  nuclei_coords =  cropped_nuclei_coords, vol_nuclei = cropped_vol_nuclei)
				sub_figs.append(temp_fig)

				list_of_all_mouse_events.append( MouseEventsFromSegmentGUI(bounding_box, list_of_mouse_events_in_cropped_ascidian ))

			
	
	fig.canvas.mpl_connect('pick_event', onpick)
	fig.canvas.mpl_connect('scroll_event', scroll_zoom)

	pylab.show()
	if matplotlib.get_backend().lower() != "GTK3Agg".lower():
		pylab.close()
		#gc.collect()
	
	print colored("=============================== END USER FEEDBACK ==============================","yellow")
	logging.info("ENDING USER FEEDBACK")



	return list_of_all_mouse_events, merge_pred.list_to_merge



