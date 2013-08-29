
import xml.etree.ElementTree as ET

from libtiff import TIFF
import StringIO
import re
import pdb



class Metadata(object):

	def __init__(self):

		self.xres = 0
		self.yres = 0
		self.zres = 0
		self.tres = 0
		
		self.numx = 0
		self.numy = 0
		self.numz = 0
		self.numt = 0

		self.numch = 0
		self.mem_ch = 0


	def metadata_etree(self):

		metadata_field =  ET.Element("metadata")
	
		ET.SubElement(metadata_field, "metafield", attrib={"name": "xres", "value" : str(self.xres)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "yres", "value" : str(self.yres)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "zres", "value" : str(self.zres)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "tres", "value" : str(self.tres)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "numx", "value" : str(self.numx)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "numy", "value" : str(self.numy)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "numz", "value" : str(self.numz)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "numt", "value" : str(self.numt)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "numch", "value" : str(self.numch)})
		ET.SubElement(metadata_field, "metafield", attrib={"name": "mem_ch", "value" : str(self.mem_ch)})


		return metadata_field

	def load_from_etree(self, metadata_field):

		for meta_field in metadata_field.iter("metafield"):
			if meta_field.get("name") == "xres":
				self.xres = float(meta_field.get("value"))
			if meta_field.get("name") == "yres":
				self.yres = float(meta_field.get("value"))
			if meta_field.get("name") == "zres":
				self.zres = float(meta_field.get("value"))
			if meta_field.get("name") == "tres":
				self.tres = float(meta_field.get("value"))
			if meta_field.get("name") == "numx":
				self.numx = int(meta_field.get("value"))			
			if meta_field.get("name") == "numy":
				self.numy = int(meta_field.get("value"))
			if meta_field.get("name") == "numz":
				self.numz = int(meta_field.get("value"))
			if meta_field.get("name") == "numt":
				self.numt = int(meta_field.get("value"))
			if meta_field.get("name") == "numch":
				self.numch = int(meta_field.get("value"))
			if meta_field.get("name") == "mem_ch":
				self.mem_ch = int(meta_field.get("value"))



	def get_meta_from_line(self, line):

		meta_needed = set(["XResolution", "YResolution", "images", "slices", "SizeC", "SizeZ", "slices"\
	                       "SizeT", "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement" ])


		for meta in meta_needed:
			# match:
			# case insensitive metadata tag
			# in the format tag = float or int
			# with : or =
			# with possible white spaces
			# possibly the number being in quotations
			matched = re.findall('(?i)%s\s*[=:]\s*"?[0-9]*\.?[0-9]+"?' % meta, line)
			if len(matched):
				# get the value out of the matched pattern
				value = re.findall("[0-9]*\.?[0-9]+", matched[0])
				value = float(value[0])
				if meta in set( ["XResolution" , "PhysicalSizeX"]):
					self.xres = value
				if meta in set( ["YResolution" , "PhysicalSizeY"]):
					self.yres = value
				if meta in set( ["ZResolution" , "PhysicalSizeZ"]):
					self.zres = value
				if meta == "TimeIncrement":
					self.tres = value
				if meta in set(["SizeZ", "slices"]):
					self.numz = int(value)
				if meta == "SizeC":
					self.numch = int(value)
				if meta == "SizeT":
					self.numt = int(value)



	def load_info_from_tif(self, filename):

		tif = TIFF.open(filename)
		buf = StringIO. StringIO(tif.info())		

		img = tif.read_image()
		self.numx = img.shape[0]
		self.numy = img.shape[1]

			
		line = buf.readline()

		while line:
			self.get_meta_from_line(line)
			line = buf.readline()
				


	def populate_metadata_boxes(self, ui):

		if not self.xres == None:
			ui.doubleSpinBox_xres.setValue(self.xres)

		if not self.yres == None:
			ui.doubleSpinBox_yres.setValue(self.yres)

		if not self.zres == None:
			ui.doubleSpinBox_zres.setValue(self.zres)

		if not self.tres == None:
			ui.doubleSpinBox_tres .setValue(self.tres)
	
		if not self.numx == None:
			ui.spinBox_numx.setValue(self.numx)

		if not self.numy == None:
			ui.spinBox_numy .setValue(self.numy)

		if not self.numz == None:
			ui.spinBox_numz.setValue(self.numz)

		if not self.numt == None:
			ui.spinBox_numt.setValue(self.numt)

		if not self.numch == None:
			ui.spinBox_numch.setValue(self.numch)

		# TODO: if number of chnnels is not set
		if not self.mem_ch == None:
			for i in xrange(self.numch):
				ui.comboBox_mem_chan.addItem(str(i))





	def load_bq_csv_file(self, file_name):

		with open(file_name) as f:
		
			info = f.readline().split(",")
			name = info[0]
			try:
				val = info[1]
			except:
				val = None

			while name and val:

				name = name.strip()
				val = val.strip()

				if name == "pixel_resolution_x":
					self.xres = float(val)
				if name == "pixel_resolution_y":
					self.yres = float(val)
				if name == "pixel_resolution_z":
					self.zres = float(val)
				if name == "pixel_resolution_t":
					self.tres = float(val)


				if name == "image_num_x":
					self.numx = int(val)
				if name == "image_num_y":
					self.numy = int(val)
				if name == "image_num_z":
					self.numz = int(val)
				if name == "image_num_t":
					self.numt = int(val)

				if name == "image_num_c":
					self.numch = int(val)

				info = f.readline().split(",")
				name = info[0]
				try:
					val = info[1]
				except:
					val = None

			self.mem_ch = 0
