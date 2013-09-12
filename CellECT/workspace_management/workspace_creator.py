from scipy import io

from libtiff import TIFF
from scipy import misc
import os
import pdb
import numpy as np
import random


from CellECT.workspace_management import cell_interior

class PrepNuclei(object):

	def __init__(self):

		self.nuclei_dict = {}

		self.config_file_dict = {}



	def prepare_nuclei(self, csv_file, ws_location):

		self.ws_location = ws_location
		self.load_nuclei(csv_file)
		self.write_mat_files()

	def prepare_random_nuclei(self, ws_location, metadata):

		self.ws_location = ws_location
		number_time_points = metadata.numt
		number_x = metadata.numx
		number_y = metadata.numy
		number_z = metadata.numz

		for time in xrange(number_time_points):
			nuclei_mat = np.zeros((100,3))
			counter = 0
			for i in xrange(100):
				x = float(random.randint(1, number_x-1))
				y = float(random.randint(1, number_y-1))
				z = float(random.randint(1, number_z-1))
				nuclei_mat[counter,:] = np.array([x,y,z])
				counter += 1
			self.nuclei_dict[time] = nuclei_mat

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

	
	def estimate_interiors(self, ws_location, metadata):

		for i in xrange(metadata.numt):
			vol = io.loadmat("%s/init_watershed_all_time_stamps/vol_t_0" % ws_location)["vol"]
	
			interior_estimator = cell_interior.CellInterior(vol)
			self.nuclei_dict[i] = interior_estimator.run_cell_centroid_estimator()

		self.write_mat_files()



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
		self.has_bg = None
		self.make_nuclei_option = None

	def set_info(self, nuclei_csv, image_stack, metadata, has_bg, make_nuclei_option):

		self.path_to_image = image_stack
		self.path_to_nuclei_csv = nuclei_csv
		self.metadata = metadata
		self.has_bg = has_bg
		self.make_nuclei_option = make_nuclei_option

	def make_dirs(self):
	
		try:

			os.mkdir(self.ws_location)

			os.mkdir("%s/config_files" % self.ws_location)
			os.mkdir("%s/init_watershed_all_time_stamps" % self.ws_location)
			os.mkdir("%s/input_slices" % self.ws_location)
			os.mkdir("%s/segs_all_time_stamps" % self.ws_location)
			os.mkdir("%s/tracker_config" % self.ws_location)
			os.mkdir("%s/training_data" % self.ws_location)
			os.mkdir("%s/temp" % self.ws_location)

		except Exception as err:
			raise err


	def prep_training_data(self):

		import CellECT
		data_location = os.system('cp %s/data/training/ascidian/* %s/training_data/' % (CellECT.__path__[0], self.ws_location))



	def write_config_files(self):

		for time in xrange(self.metadata.numt):
			config_file_name = "%s/config_files/timestamp_%d.cnf" % (self.ws_location, time)
			with open(config_file_name, "w") as f:
				f.write("volume_mat_path = init_watershed_all_time_stamps/vol_t_%d.mat\n" % time)
				f.write("volume_mat_var = vol\n")
				f.write("first_seg_mat_path =  init_watershed_all_time_stamps/init_ws_%d.mat\n" % time)
				f.write("first_seg_mat_var = ws\n")
				f.write("nuclei_mat_path =  init_watershed_all_time_stamps/time_stamp_%d_nuclei.mat\n" % time)
				f.write("nuclei_mat_var = seeds\n")
				f.write("training_vol_mat_path =  training_data/positive_example.mat\n")
				f.write("training_vol_mat_var = vol\n")
				f.write("training_vol_nuclei_mat_path = training_data/positive_example.mat\n")
				f.write("training_vol_nuclei_mat_var = seeds\n")
				f.write("training_positive_seg_mat_path = training_data/positive_example.mat\n")
				f.write("training_positive_seg_mat_var = label_map\n")
				f.write("training_positive_labels_mat_path = training_data/positive_example.mat\n")
				f.write("training_positive_labels_mat_var = labels\n")
				f.write("training_negative_seg_mat_path = training_data/negative_example.mat\n")
				f.write("training_negative_seg_mat_var = L\n")
				f.write("training_negative_labels_mat_path = training_data/negative_example.mat\n")
				f.write("training_negative_labels_mat_var = labels\n")
				f.write("save_location_prefix = segs_all_time_stamps/timestamp_%d_\n" % time)
				f.write("has_bg = %d\n" % int(self.has_bg))
				f.write("use_size = 1\n")
				f.write("use_border_intensity = 1\n")
				f.write("use_border_distance = 0\n")



	def build_workspace(self, ws_location, progressBar):
		
		try:
			self.ws_location = ws_location
			self.make_dirs()
			progressBar.setValue(2)

			self.image = PrepImage()
			self.image.set_info(self.path_to_image, self.metadata)
			self.image.prep_image(self.ws_location, progressBar)

			self.nuclei = PrepNuclei()
			if len(self.path_to_nuclei_csv):
				self.nuclei.prepare_nuclei( self.path_to_nuclei_csv, self.ws_location  )
			else:
				if self.make_nuclei_option == "use_random":
					self.nuclei.prepare_random_nuclei (self.ws_location, self.metadata )
				if self.make_nuclei_option == "use_estimate":
					self.nuclei.estimate_interiors(self.ws_location, self.metadata)
					
			

			self.prep_training_data()

			self.write_config_files()

		except Exception as err:
			raise err




