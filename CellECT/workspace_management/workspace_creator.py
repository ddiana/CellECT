from scipy import io

from libtiff import TIFF
from scipy import misc
import os
import pdb
import numpy as np

class PrepNuclei(object):

	def __init__(self):

		self.nuclei_dict = {}

		self.config_file_dict = {}



	def prepare_nuclei(self, csv_file, ws_location):

		self.ws_location = ws_location
		self.load_nuclei(csv_file)
		self.write_mat_files()



	def write_mat_files(self):

		for time in self.nuclei_dict.keys():
			nuclei_mat = np.zeros((len(self.nuclei_dict[time]), 3))
			counter = 0
			for nuclei_coords in self.nuclei_dict[time]:
				nuclei_mat[counter, :] = np.array(nuclei_coords)
				counter += 1
			
			file_name = "%s/init_watershed_all_time_stamps/time_stamp_%d_nuclei.mat" % (self.ws_location, time)
			io.savemat( file_name, {"seeds": nuclei_mat.T})

		


	def load_nuclei(self, csv_file):


		with open(csv_file) as f:
			# skip the first line (header)
			line = f.readline()
			line = f.readline()
			while line:
				
				items = [float(item.strip()) for item in line.split(",")]
				time = int(items[3])
				if not self.nuclei_dict.has_key(time):
					self.nuclei_dict[time] = []
	
				self.nuclei_dict[time].append([items[0:3]])

				line = f.readline()

				
	


class PrepImage(object):

	def __init__(self):

		self.img_location = None
		self.metadata = None


	def set_info(self, img_location, metadata):
		self.metadata = metadata
		self.img_location = img_location



	def prep_image(self, ws_location, progressBar):

		tif = TIFF.open(self.img_location)

		counter = 0
	
		z_counter = 0

		mat = np.zeros((self.metadata.numx, self.metadata.numy, self.metadata.numz))
		time_counter = 0

		for img in tif.iter_images():

			filename = "%s/input_slices/%d.tif" % (ws_location, counter+1)
			misc.imsave(filename, img)

			should_add_slice = False

			if 	counter % self.metadata.numch == self.metadata.mem_ch:
				should_add_slice = True


			if should_add_slice:
				mat[:,:,z_counter] = img
				z_counter += 1
				if z_counter >= self.metadata.numz:
					z_counter = 0

					io.savemat("%s/init_watershed_all_time_stamps/vol_t_%d.mat" % (ws_location, time_counter), {"vol": mat})
					time_counter += 1

					progressBar.setValue( int(time_counter/ float(self.metadata.numt) * 100) )

			counter += 1

	
		




class WorkspaceCreator(object):


	def __init__(self):

		self.path_to_nuclei_csv = None
		self.path_to_image = None
		self.metadata = None


	def set_info(self, nuclei_csv, image_stack, metadata):

		self.path_to_image = image_stack
		self.path_to_nuclei_csv = nuclei_csv
		self.metadata = metadata
		

	def make_dirs(self):
	
		os.mkdir(self.ws_location)

		os.mkdir("%s/config_files" % self.ws_location)
		os.mkdir("%s/init_watershed_all_time_stamps" % self.ws_location)
		os.mkdir("%s/input_slices" % self.ws_location)
		os.mkdir("%s/segs_all_time_stamps" % self.ws_location)
		os.mkdir("%s/tracker_config" % self.ws_location)
		os.mkdir("%s/training_data" % self.ws_location)
		os.mkdir("%s/temp" % self.ws_location)


	def prep_training_data(self):

		import CellECT
		data_location = os.system('cp %s/data/training/ascidian/* %s/training_data/' % (CellECT.__path__[0], self.ws_location))



	def build_workspace(self, ws_location, progressBar):
		
		self.ws_location = ws_location
		self.make_dirs()
		progressBar.setValue(2)

		self.nuclei = PrepNuclei()
		self.nuclei.prepare_nuclei( self.path_to_nuclei_csv, self.ws_location  )
		

		self.image = PrepImage()
		self.image.set_info(self.path_to_image, self.metadata)
		self.image.prep_image(self.ws_location, progressBar)

		self.prep_training_data()





