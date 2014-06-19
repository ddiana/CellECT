# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
import os
import pdb
import os.path
import xml.etree.ElementTree as ET

# Imports from this project
from CellECT.workspace_management import metadata as md


class WorkSpaceData(object):

	def __init__ (self):

		self.metadata = md.Metadata()
		self.available_segs = []
		self.workspace_location = None




	def append_workspace(self,temp_ws):


		if self.metadata.numx == temp_ws.metadata.numx and \
           self.metadata.numy == temp_ws.metadata.numy and \
           self.metadata.numz == temp_ws.metadata.numz and \
           self.metadata.numch == temp_ws.metadata.numch and \
           self.metadata.memch == temp_ws.metadata.memch and \
           self.metadata.xres == temp_ws.metadata.xres and \
           self.metadata.yres == temp_ws.metadata.yres and \
           self.metadata.zres == temp_ws.metadata.zres and \
           self.metadata.tres == temp_ws.metadata.tres:

			time_offset = self.metadata.numt
			self.metadata.numt += temp_ws.metadata.numt
			for item in temp_ws.available_segs:
				self.available_segs.add(item + time_offset)


		else:
			raise Exception("ValueError", "The metadata does not match.")

	def set_location(self, filename):

		self.workspace_location = os.path.dirname(filename)
		


	def load_metadata(self, filename):

		self.metadata.load_bq_csv_file(filename)

		self.set_location(filename)


	def get_available_segs(self):

		self.available_segs = set()
		for i in xrange (self.metadata.numt):
			file_name = os.path.join(self.workspace_location, "segs_all_time_stamps","timestamp_%d_label_map.mat" % i)
   			if os.path.exists(file_name):
				self.available_segs.add(i)


	def load_workspace(self, location):

		tree = ET.parse(location)
		root = tree.getroot()
		metadata_field = root.find("metadata")
		self.metadata.load_from_etree(metadata_field)


	def write_xml(self):
	
		root = ET.Element("CellECT_workspace")

		metadata_field = self.metadata.metadata_etree()
	
		root.append(metadata_field)
	
		tree = ET.ElementTree(root)

		file_name = os.path.join(self.workspace_location,"workspace_data.cws")
		tree.write(file_name)

		



