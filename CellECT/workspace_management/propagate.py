import xml.etree.ElementTree as ET
import pdb
import re
import numpy as np
from scipy import io
from os import path
import os
from scipy.ndimage.morphology import binary_erosion

from CellECT.seg_tool.seg_utils.union_find import UnionFind

class PreparePropagateInput(object):

	def __init__(self, ws_data, tp, direction):

		self.ws_data = ws_data
		self.time_point = tp
		self.direction = direction
		self.prepare_propagate()



	def prepare_propagate(self):
		
		# direction = +1 (for backward)
		# direction = -1 (for forward)


		# load only the head nuclei from the time point ahead, if there is such saved output

		file_name = "%s/segs_all_time_stamps/timestamp_%d_segment_props.xml" % (self.ws_data.workspace_location ,self.time_point+ self.direction)

		if not path.isfile(file_name):
			raise Exception("No saved data to propagate.")
		
		try:
			nuclei_pts = self.load_inner_pts_from_xml(file_name)
		except Exception as err:
			raise err

		if len(nuclei_pts) == 0:
			raise Exception("No information in saved data.")

		bg_seeds = set()
		bg_seeds = []
		# if propagate segmentation result exists from before, delete it, so that seg tool recomputes.
	
		prev_seg_file_name = "%s/init_watershed_all_time_stamps/init_ws_%d_propagate.mat" % (self.ws_data.workspace_location ,self.time_point)
		os.system('[ -f "%s" ] && rm %s' % (prev_seg_file_name, prev_seg_file_name))


#		try:
#			file_name = "%s/segs_all_time_stamps/timestamp_%d_bg_seeds.xml" % (self.ws_data.workspace_location ,self.time_point+ self.direction)
#			bg_seeds = self.load_bg_seeds_from_xml(file_name)
#			bg_seeds = list(bg_seeds)
#		except:
#			print "Could not propagate user saved background seeds. Trying default background seeds."

#			try:		
#				file_name = "%s/segs_all_time_stamps/timestamp_%d_bg_seeds.mat" % (self.ws_data.workspace_location ,self.time_point+ self.direction)
#				import scipy.io as sio
#				bg_seeds = sio.loadmat(file_name)["bg_seeds"]
#				bg_seeds = [x for x in bg_seeds]
#			except:
#				print "Could not propagate any background seeds. Continuing without background seeds."
			

		#nuclei_pts = self.prune_nuclei(nuclei_pts, union_find)

		file_name = "%s/init_watershed_all_time_stamps/time_stamp_%d_nuclei_propagate.mat" % (self.ws_data.workspace_location, self.time_point)

		self.write_mat_file(file_name, nuclei_pts)


		bg_mask = []

		try:
			file_name = "%s/segs_all_time_stamps/timestamp_%d_label_map.mat" % (self.ws_data.workspace_location ,self.time_point+ self.direction)
			bg_mask = self.load_bg_mask_from_seg(file_name)
		except:
			pass

		file_name = "%s/init_watershed_all_time_stamps/time_stamp_%d_bg_seeds_propagate.mat" % (self.ws_data.workspace_location, self.time_point)

		io.savemat(file_name, {"seeds":bg_seeds, "bg_mask": bg_mask})

		self.write_vol_file()



	def write_vol_file(self,):


		next_seg = io.loadmat("%s/segs_all_time_stamps/timestamp_%d_label_map.mat" % (self.ws_data.workspace_location, self.time_point + self.direction))["ws"]
		current_vol = io.loadmat("%s/init_watershed_all_time_stamps/vol_t_%d.mat"  % (self.ws_data.workspace_location, self.time_point))["vol"]

		from scipy.ndimage.morphology import distance_transform_edt

		dt = distance_transform_edt(next_seg>1)
		dt = 1 - dt / float(dt.max())
		dt = dt / 2. + 0.5
		current_vol = current_vol * dt

		io.savemat("%s/init_watershed_all_time_stamps/vol_t_%d_propagate.mat"% (self.ws_data.workspace_location, self.time_point), {"vol":current_vol})




	def write_mat_file(self, file_name, nuclei_pts):

		nuclei_mat = np.zeros((len(nuclei_pts), 3))
		counter = 0
		for nuclei_coords in nuclei_pts:
			nuclei_mat[counter, :] = np.array(nuclei_coords)
			counter += 1
			
		io.savemat( file_name, {"seeds": nuclei_mat.T})




	def prune_nuclei(self, nuclei_pts, union_find):

		
		final_set_nuclei = []

		for i in xrange(len(union_find.parents)):
			if union_find.find(i) == i:	
				final_set_nuclei.append(nuclei_pts[i])

		return final_set_nuclei



	def load_bg_seeds_from_xml(self, file_name):
	
		try:
			tree = ET.parse(file_name)
		except IOError as err:
			err.message = "Could not open nuclei xml file at %s" % file_name
			raise err

		root = tree.getroot()
		bg_seeds = eval(tree.findall("list_of_seeds")[0].text)
	
		return bg_seeds

	def load_bg_mask_from_seg(self, file_name):

		ws = io.loadmat(file_name)["ws"]

		bg_mask = (ws == 1).astype(ws.dtype)
		bg_mask = binary_erosion(bg_mask, iterations = 15, mask = bg_mask)

		return bg_mask

			


	def load_inner_pts_from_xml(self, file_name):

		"Load nuclei collection from xml file."
	
		try:
			tree = ET.parse(file_name)
		except IOError as err:
			err.message = "Could not open nuclei xml file at %s" % file_name
			raise err


		root = tree.getroot()

		nuclei_list = []



		seg_list_root = root.find('list_of_segments')

		for segment in seg_list_root:
			feats = segment.find('feature_dictionary')
			for feature in feats:
				if feature.attrib['name'] == "inner_point":
					coords = eval(feature.text)
					nuclei_list.append(coords)

#		for child in root:
#			if child.tag == "nucleus":
#				x = int(child.attrib["x"])
#				y = int(child.attrib["y"])
#				z = int(child.attrib["z"])
#				index = int(child.attrib["index"])
#				nuclei_list.append((x,y,z))

#		union_find_field = tree.findall("union_find")
#		parents_string = union_find_field[0][0].text
#		parents_string = re.findall("(\d+)", parents_string)
#		parents = [int(val) for val in parents_string]

#		set_size_string = union_find_field[0][1].text
#		set_size_string = re.findall("(\d+)", set_size_string)
#		set_size = [int(val) for val in set_size_string]

#		union_find = UnionFind(0)
#		union_find.parents = parents
#		union_find.set_size = set_size


		return nuclei_list  #, union_find



			

		

