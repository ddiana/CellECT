# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
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
		self.memch = 0
		
		self.num_pages = 0


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
		ET.SubElement(metadata_field, "metafield", attrib={"name": "memch", "value" : str(self.memch)})


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
			if meta_field.get("name") == "memch":
				self.memch = int(meta_field.get("value"))

		if self.numt ==0:
			self.numt = 1

		if self.numz ==0:
			self.numz =1

		if self.numch == 0:
			self.numch = 1

		self.memch = 0



	def load_bq_xml_file(self, filename):
	# open xml like text file, and read line by line just like tif metadata

	
		with open(filename, "r") as input_file:

			line = input_file.readline()
			while line:
				self.get_meta_from_line(line)
				line = input_file.readline()

		if self.numt ==0:
			self.numt = 1

		if self.numz ==0:
			self.numz =1

		if self.numch == 0:
			self.numch = 1

		self.memch = 0


		

	def get_meta_from_line(self, line):

		# TODO unify metadata in a dictionary

		meta_needed = set(["XResolution", "YResolution", "images", "slices", "SizeC", "SizeZ", "slices",\
	                       "SizeT", "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ", "TimeIncrement",\
	                       "pixel_resolution_x", "pixel_resolution_y", "pixel_resolution_z", "pixel_resolution_t",\
	                       "image_num_c", "image_num_z", "image_num_t", "image_num_x", "image_num_y" ])



		for meta in meta_needed:
			# match:
			# case insensitive metadata tag
			# in the format tag = float or int
			# with : or =
			# with possible white spaces
			# possibly the number being in quotations
			matched = re.findall('(?i)%s\s*[=:]\s*("?[0-9a-zA-Z]*\.?[0-9a-zA-Z]+"?)' % meta, line)

			# if nothing found, this can be an xml file, look for xml format:
			matched = re.findall('name="(?i)%s"\s*value=("?[0-9a-zA-Z]*\.?[0-9a-zA-Z]+"?)' % meta, line)


			# = re.search('[a-zA-Z]',matched[0])
			if len(matched) :
				# get the value out of the matched pattern

				for match in matched:
					#value = re.findall("[0-9]*\.?[0-9]+", match)
					value = match.strip("'")
					value = value.strip('"')

					try:
						value = float(value)
					except:
						continue
	
					if meta in set( ["XResolution" , "PhysicalSizeX", "pixel_resolution_x"]):
						self.xres = value
					if meta in set( ["YResolution" , "PhysicalSizeY", "pixel_resolution_y"]):
						self.yres = value
					if meta in set( ["ZResolution" , "PhysicalSizeZ", "pixel_resolution_z"]):
						self.zres = value
					if meta in set([ "TimeIncrement", "pixel_resolution_t"]):
						self.tres = value
					if meta in set(["SizeZ", "slices", "image_num_z"]):
						if value > self.numz:   # because some datasets have two value for SizeZ
							self.numz = int(value)
					if meta in set(["image_num_ch","SizeC"]):
						self.numch = int(value)
					if meta in set(["SizeT", "image_num_t"]):
						self.numt = int(value)

					# only check x and y if reading metadata from xml (bq), otherwise get it from tif
					if meta in set(["image_num_x"]):
						self.numx = int(value)
					if meta in set(["image_num_y"]):
						self.numy = int(value)

		if self.numt ==0:
			self.numt = 1

		if self.numz ==0:
			self.numz =1

		if self.numch == 0:
			self.numch = 1


		self.memch = 0





	def load_info_from_tif(self, filename):

		tif = TIFF.open(filename)
		buf = StringIO. StringIO(tif.info())		

		img = tif.read_image()

		for image in tif.iter_images():
			self.num_pages += 1

		print self.num_pages
		self.numx = img.shape[0]
		self.numy = img.shape[1]

			
		line = buf.readline()

		while line:
			self.get_meta_from_line(line)
			line = buf.readline()

		if self.numt ==0:
			self.numt = 1

		if self.numz ==0:
			self.numz =1

		if self.numch == 0:
			self.numch = 1
				
		self.memch = 0



	def populate_metadata_boxes(self, ui):


		ui.doubleSpinBox_xres.setValue(self.xres)

		ui.doubleSpinBox_yres.setValue(self.yres)

		ui.doubleSpinBox_zres.setValue(self.zres)

		ui.doubleSpinBox_tres .setValue(self.tres)
	
		ui.spinBox_numx.setValue(self.numx)

		ui.spinBox_numy .setValue(self.numy)

		ui.spinBox_numz.setValue(self.numz)

		ui.spinBox_numt.setValue(self.numt)

		ui.spinBox_numch.setValue(self.numch)


		# change in this value will also set the list for membrane channel.

#		# TODO: if number of chnnels is not set
#		if not self.memch == None:
#			for i in xrange(self.numch):
#				ui.comboBox_mem_chan.addItem(str(i))





	def save_csv_file(self,file_name):

		try:
			with open(file_name, "w") as f:

				f.write("pixel_resolution_x, %f\n" % self.xres)
				f.write("pixel_resolution_y, %f\n" % self.yres)
				f.write("pixel_resolution_z, %f\n" % self.xres)
				f.write("pixel_resolution_t, %f\n" % self.tres)
				f.write("image_num_x, %d\n" % self.numx)
				f.write("image_num_y, %d\n" % self.numy)
				f.write("image_num_z, %d\n" % self.numz)
				f.write("image_num_t, %d\n" % self.numt)
				f.write("image_num_c, %d\n" % self.numch)
				f.write("memch, %d\n" % self.memch)

		except Exception as err:
			raise err


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

				if name == "memch":
					self.memch = int(val)


				info = f.readline().split(",")
				name = info[0]
				try:
					val = info[1]
				except:
					val = None

			if self.memch == None:
				self.memch = 0
