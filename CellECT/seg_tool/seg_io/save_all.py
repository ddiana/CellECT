# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import scipy as sp
from scipy import io
import pdb
from termcolor import colored
import sys
import logging
import numpy as np

from scipy import ndimage

# Import from this project
from CellECT.seg_tool.seg_io import save_xml 
import CellECT.seg_tool.globals

"""
Functions to save the current status.
"""

def save_current_status(nuclei_collection, seed_collection,segment_collection, seed_segment_collection, label_map, bg_seeds, bg_prior):
	
	"Save XML and MAT files."

	logging.info("Saving current status.")
	
	try:

		segment_collection.make_contours_for_all_segments(label_map)

		print colored("Saving XML for nuclei...", "cyan")
		save_xml.save_xml_file_nuclei(nuclei_collection)
	
		print colored("Saving XML for seeds...", "cyan")
		save_xml.save_xml_file_seeds(seed_collection)	
	
		print colored("Saving XML for segment properties...", "cyan")
		save_xml.save_xml_file_segment_props(segment_collection)
	
		print colored("Saving XML for seed-segment properties...", "cyan")	
		save_xml.save_xml_file_seed_segment_props(seed_segment_collection)

		print colored("Saving XML for background seeds...", "cyan")	
		save_xml.save_xml_file_bg_seeds(bg_seeds)
	
		if not CellECT.seg_tool.globals.DEFAULT_PARAMETER["bisque"]:

			# removing background segments border if necessary

			print colored("Saving label map in MAT file...", "cyan")
			save_seg_to_mat(label_map, bg_prior)
	
			print colored("Saving segmentation slices as PNG files...", "cyan")
			save_seg_slices(label_map)
	
	except Exception as err:

		print colored("Error: %s" % err.message, "red")
		print colored(err, "red")
		logging.exception(err)
		logging.exception(err.message)
		sys.exit()


	print colored("Done saving!", "cyan")





def save_seg_to_mat(watershed, bg_prior):

	"Save label map to .mat file."

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "label_map.mat"

	try:
		if bg_prior is not None:
			io.savemat(file_name, {"ws": watershed, "bg_mask": bg_prior})
		else:
			io.savemat(file_name, {"ws": watershed})
	except Exception as err:
		err.message = "Error saving variable in  file %s" % file_name
		raise err

	print "....... Saved label mat at:", file_name

	
	

def save_seg_slices(seg):

	"Save segmentations lices in png files."

	for i in xrange (seg.shape[2]):
		seg_slice = sp.misc.toimage(seg[:,:,i], high = seg[:,:,i].max(), low = seg[:,:,i].min(), mode = 'I')
		file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "z_" + str(i) + "_seg.png"
		try:			
			seg_slice.save(file_name)
		except IOError as err:
			err.message = "Error saving image file at %s" % file_name
			raise err
		

	file_name = CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"] + "z_"+ "***" + "_seg.png"	
	print "....... Saved label map PNG slices at:", CellECT.seg_tool.globals.DEFAULT_PARAMETER["save_location_prefix"]


