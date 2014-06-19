# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from scipy import io
from libtiff import TIFF
from scipy import misc
import os
import pdb
import numpy as np
import os.path
import random

# Imports from this project
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
			
			file_name =  os.path.join(self.ws_location,  "init_watershed_all_time_stamps","time_stamp_%d_nuclei.mat" % time)
			io.savemat( file_name, {"seeds": nuclei_mat.T})

		

	def use_no_nuclei(self, ws_location, metadata):
		
		self.ws_location = ws_location
		number_time_points = metadata.numt
		number_x = metadata.numx
		number_y = metadata.numy
		number_z = metadata.numz

		for time in xrange(number_time_points):
#			nuclei_mat = np.zeros((1,3))
#			x = float(round(number_x / 2))
#			y = float(round(number_y / 2))
#			z = float(round(number_z / 2))
#			nuclei_mat[0,:] = np.array([x,y,z])
			self.nuclei_dict[time] = [] #nuclei_mat


		self.write_mat_files()

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
	
				self.nuclei_dict[time].append([items[1], items[0], items[2]])

				line = f.readline()

	
	def estimate_interiors(self, ws_location, metadata):

		for i in xrange(metadata.numt):
			vol = io.loadmat( os.path.join(ws_location, "init_watershed_all_time_stamps","vol_t_0"))["vol"]
	
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
	
		z_counter_mem = 0
		z_counter_nuc = 0


		mat = np.zeros((self.metadata.numx, self.metadata.numy, self.metadata.numz))
		time_counter = 0


		mat_nuclei = None
		if self.metadata.numch >1:
			mat_nuclei = np.zeros_like(mat)
		

		for img in tif.iter_images():

			filename = os.path.join(ws_location, "input_slices", "%d.tif" %( counter+1))
			misc.imsave(filename, img)

			membrane_slice = False


			if 	counter % self.metadata.numch == self.metadata.memch:
				membrane_slice = True

			if membrane_slice:
				mat[:,:,z_counter_mem] = img
				z_counter_mem += 1

			else:
				mat_nuclei[:,:,z_counter_nuc] = img
				z_counter_nuc += 1


			if z_counter_mem >= self.metadata.numz:
				z_counter_mem = 0
				maxim = mat.max()
				if maxim:
					mat *= 255./maxim
				io.savemat(os.path.join(ws_location, "init_watershed_all_time_stamps","vol_t_%d.mat" % time_counter), {"vol": mat})

			if z_counter_nuc >= self.metadata.numz:
				z_counter_nuc = 0
				maxim = mat_nuclei.max()
				if maxim:
					mat_nuclei *= 255./maxim
				io.savemat(os.path.join(ws_location, "init_watershed_all_time_stamps", "vol_nuclei_t_%d.mat" % time_counter), {"vol_nuclei": mat_nuclei})

			if z_counter_mem == 0 and z_counter_nuc == 0:
				time_counter += 1
	

				progressBar.setValue( int(time_counter/ float(self.metadata.numt) * 100) )



#			if membrane_slice:
#				mat[:,:,z_counter] = img
#				z_counter += 1
#				if z_counter >= self.metadata.numz:
#					z_counter = 0

#					io.savemat("%s/init_watershed_all_time_stamps/vol_t_%d.mat" % (ws_location, time_counter), {"vol": mat})
#					time_counter += 1

#					progressBar.setValue( int(time_counter/ float(self.metadata.numt) * 100) )

#			else:
#				mat_nuclei[:,:,z_counter] = img

#				if z_counter >= self.metadata.numz:
#					z_counter = 0

#					io.savemat("%s/init_watershed_all_time_stamps/vol_t_%d.mat" % (ws_location, time_counter), {"vol": mat})
#					time_counter += 1

#					progressBar.setValue( int(time_counter/ float(self.metadata.numt) * 100) )

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

			os.mkdir(os.path.join (self.ws_location,"config_files"))
			os.mkdir(os.path.join (self.ws_location,"init_watershed_all_time_stamps"))
			os.mkdir(os.path.join (self.ws_location,"input_slices"))
			os.mkdir(os.path.join (self.ws_location,"segs_all_time_stamps" ))
			os.mkdir(os.path.join (self.ws_location,"tracker_config"))
			os.mkdir(os.path.join (self.ws_location,"training_data"))
			#os.mkdir("%s/temp" % self.ws_location)

		except Exception as err:
			raise err


	def prep_training_data(self):

		import CellECT
		try:
			training_data_location = os.path.join(CellECT.__path__[0],"data","training","ascidian","positive_example.mat")
			io.loadmat(training_data_location)
			training_data_location = os.path.join(CellECT.__path__[0],"data","training","ascidian","negative_example.mat")
			io.loadmat(training_data_location)
			data_location = os.system('cp %s* %s' % (os.path.join(CellECT.__path__[0], "data", "training","ascidian", ""), os.path.join(self.ws_location, "training_data", "")))
		except IOError as err:
			err.message = "no_cellness"
			raise err





	def write_config_files(self, time_range = None):


		if time_range == None:
			time_range = xrange(self.metadata.numt)

		for time in time_range:

			for filetype in ("", "_propagate"):
				config_file_name = os.path.join(self.ws_location, "config_files","timestamp_%d%s.cnf" % ( time, filetype))
				with open(config_file_name, "w") as f:
					path_var = os.path.join("init_watershed_all_time_stamps","vol_t_%d%s.mat" % (time,filetype))
					f.write("volume_mat_path = %s\n" % path_var)
					f.write("volume_mat_var = vol\n")
					path_var = os.path.join("init_watershed_all_time_stamps","vol_nuclei_t_%d.mat" % time)
					f.write("volume_nuclei_mat_path = %s\n" % path_var)         
					f.write("volume_nuclei_mat_var = vol_nuclei\n")
					path_var = os.path.join("init_watershed_all_time_stamps","init_ws_%d%s.mat" % (time,filetype))
					f.write("first_seg_mat_path = %s\n" % path_var)      
					f.write("first_seg_mat_var = ws\n")
					path_var = os.path.join("init_watershed_all_time_stamps","time_stamp_%d_nuclei%s.mat" % (time, filetype))
					f.write("nuclei_mat_path = %s\n" % path_var)        
					f.write("nuclei_mat_var = seeds\n")
					path_var = os.path.join("init_watershed_all_time_stamps","time_stamp_%d_bg_seeds%s.mat" % (time, filetype))
					f.write("bg_seeds_path = %s\n" % path_var)           
					f.write("bg_seeds_var = seeds\n")
					path_var = os.path.join("training_data","positive_example.mat")
					f.write("training_vol_mat_path = %s\n" % path_var)         
					f.write("training_vol_mat_var = vol\n")
					path_var = os.path.join("training_data","positive_example.mat")
					f.write("training_vol_nuclei_mat_path = %s\n" % path_var)        
					f.write("training_vol_nuclei_mat_var = seeds\n")
					path_var = os.path.join("training_data","positive_example.mat")
					f.write("training_positive_seg_mat_path = %s\n" % path_var)        
					f.write("training_positive_seg_mat_var = label_map\n")
					path_var = os.path.join("training_data","positive_example.mat")
					f.write("training_positive_labels_mat_path = %s\n" % path_var)    
					f.write("training_positive_labels_mat_var = labels\n")
					path_var = os.path.join("training_data","negative_example.mat")
					f.write("training_negative_seg_mat_path = %s\n" % path_var)        
					f.write("training_negative_seg_mat_var = L\n")
					path_var = os.path.join("training_data","negative_example.mat")
					f.write("training_negative_labels_mat_path = %s\n" % path_var)        
					f.write("training_negative_labels_mat_var = labels\n")
					path_var = os.path.join("segs_all_time_stamps","timestamp_%d_" % time)
					f.write("save_location_prefix = %s\n" % path_var)           
					f.write("has_bg = %d\n" % int(self.has_bg))
					f.write("use_size = 1\n")
					f.write("use_border_intensity = 1\n")
					f.write("use_border_distance = 0\n")
					f.write("use_dist_from_margin = 1\n")				
					f.write("x_res = %f\n" % self.metadata.xres)
					f.write("y_res = %f\n" % self.metadata.yres)
					f.write("z_res = %f\n" % self.metadata.zres)
					path_var = os.path.join("init_watershed_all_time_stamps","time_stamp_%d.csv" % time)
					f.write("ap_axis_file = %s\n" % path_var)          

	def build_workspace(self, ws_location, progressBar):
		# called from new_workspace.py
		
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
				if self.make_nuclei_option == "no_nuclei":
					self.nuclei.use_no_nuclei(self.ws_location, self.metadata)
					
			

			self.prep_training_data()

			self.write_config_files()
		except IOError as err:
			if err.message == "no_cellness":
				err.message = "No training data available. You will only be able to run without cellness metric."
				self.write_config_files()
			raise err

		except Exception as err:
			raise err




