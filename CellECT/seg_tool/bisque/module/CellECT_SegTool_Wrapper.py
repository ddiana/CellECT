# BISQUE wrapper for CellECT_seg_tool

import sys
from lxml import etree as ET
from bq.api import BQSession, BQTag
import numpy as np
from scipy import io
import parser
import logging
import os
import pdb
from libtiff import TIFF
import cv2
import optparse
from bq.api.util import AttrDict



class CellECT_SegTool_Wrapper(object):
	""" Wrapper class for CellECT_seg_tool for Bisque
	"""


	def create_workspace(self, workspace_path):
		""" Create the directory structure of a temporary workspace in which to 
			prepare the input files CellECT_seg_tool needs.
		"""

		if not os.path.exists(workspace_path + "/config_files"):
			try:
				os.makedirs(workspace_path + "/config_files")
				os.makedirs(workspace_path + "/init_watershed_all_time_stamps")
				os.makedirs(workspace_path + "/input_slices")
				os.makedirs(workspace_path + "/segs_all_time_stamps")
				os.makedirs(workspace_path + "/tracker_config")
				os.makedirs(workspace_path + "/training_data")
				os.makedirs(workspace_path + "/temp")
			except IOError as err:
				err.message ("Could not create workspace directory")
				raise err

		print "Successfully created workspace at %s" % workspace_path



	def get_image(self):

		""" Fetch one time stamp of the image the user selected.
		"""

		nd3d_mex_node = self.mex_etree.findall('.//tag[@name="inputs"]/tag[@name="input_mex"]')
		self.nd3d_mex_url = nd3d_mex_node[0].attrib["value"]
		nd3d_mex_etree = self.bqsession.fetchxml(self.nd3d_mex_url + "?view=deep,clean")
		image_url_node = nd3d_mex_etree.findall('.//tag[@name="inputs"]/tag[@name="resource_url"]')
		self.image_url = image_url_node[0].attrib["value"]

		self.image = self.bqsession.load(self.image_url)

		image_meta_etree = ET.fromstring(self.image.pixels().meta().fetch())

		number_time_stamps =  int(image_meta_etree.findall('.//tag[@name="image_num_t"]')[0].attrib["value"])
		number_channels = int(image_meta_etree.findall('.//tag[@name="image_num_c"]')[0].attrib["value"])
		number_z_per_time = int(image_meta_etree.findall('.//tag[@name="image_num_z"]')[0].attrib["value"])
		number_x = int(image_meta_etree.findall('.//tag[@name="image_num_x"]')[0].attrib["value"])
		number_y = int(image_meta_etree.findall('.//tag[@name="image_num_y"]')[0].attrib["value"])


		self. input_image = np.zeros((number_y, number_x, number_z_per_time))


		for i in xrange(number_z_per_time):
			temp_img_pixels = self.image.pixels()
			temp_img_pixels.slice(z=str(i),t=str(self.time_stamp+1))
			temp_img_pixels.ops.append('depth=8,d')
			temp_img_pixels.ops.append('remap=%s'% self.membrane_channel)
			temp_img_pixels.format('ome-tiff')
			temp_img = temp_img_pixels.fetch()
			with open('temp.ome.tiff', "wb") as f:
				f.write(temp_img)
			self.input_image[:,:,i] = cv2.imread('temp.ome.tiff')[:,:,0]



			
		 

	def get_ND3D_output(self):

		""" Fetch the nuclei detector output for the timestamp the user selected.
		"""

		self.nd3d_mex_etree = self.bqsession.fetchxml(self.nd3d_mex_url+'?view=deep')
		vertex_list = self.nd3d_mex_etree.findall('.//point[@name="centroid"]/vertex')
		vertex_list = filter(lambda x: float(x.attrib['t']) == self.time_stamp, vertex_list)
		self.nuclei_list = [[int(float(vertex.attrib['y'])), int(float(vertex.attrib['x'])), int(float(vertex.attrib['z']))] for  vertex in vertex_list ]
		self.nuclei_list = np.array(self.nuclei_list).T

	

	def setup_input(self):

		""" Start the BQ session, get the information the user selected in the UI.
		"""



		self.bqsession = BQSession().init_mex(self.mex_url, self.access_token)
		self.mex_etree = self.bqsession.fetchxml(self.mex_url+'?view=deep')


		try:
			self.time_stamp = int(self.mex_etree.findall('.//tag[@name="inputs"]/tag[@name="time_stamp"]')[0].attrib["value"])
		except:
			self.time_stamp = 0

		try:
			self.membrane_channel = int(self.mex_etree.findall('.//tag[@name="inputs"]/tag[@name="membrane_channel"]')[0].attrib["value"])
		except:
			self.membrane_channel = 1


		try:
			self.has_background = self.mex_etree.findall('.//tag[@name="inputs"]/tag[@name="has_background"]')[0].attrib["value"]
			if self.has_background == "True":
				self.has_background = True
		except:
			self.has_background = False

		self.cellness_metric = self.mex_etree.findall('.//tag[@name="inputs"]/tag[@name="cellness_metric"]')[0].attrib["value"]




	def attach_results(self):

		""" Once CellECT_seg_tool finished running, append the bisque XML to the
			mex's' outputs section.
		"""
		
		
		ET.SubElement(self.mex_etree, "tag", attrib={"name":"outputs"})
		node_to_attach_gobjects = ET.SubElement(self.mex_etree.find('.//tag[@name="outputs"]'), "tag", attrib={"name":"Segmented_Image", "type": "image", "value":self.image_url})
				
		self.results_xml = ET.parse('temp_workspace/segs_all_time_stamps/timestamp_0_bisque.xml')
		node_to_attach_gobjects.append(self.results_xml.getroot())

		self.bqsession.postxml(self.mex_url, self.mex_etree)
		


	def run_first_watershed(self):

		""" First watershed segmentation using nuclear detector output as seeds
		"""
	
		from CellECT.seg_tool.seg_io.load_parameters import abs_path_to_workspace
		path_to_mat_input = abs_path_to_workspace('temp_workspace/config_files/config_file.cfg')+"/temp/watershed_input.mat"
		path_to_mat_result = abs_path_to_workspace('temp_workspace/config_files/config_file.cfg')+"/temp/watershed_result.mat"

		import CellECT
		matlab_m_file_path = CellECT.__path__[0] + "/utils"

		try:
			io.savemat(path_to_mat_input, {"vol": self.input_image, "seeds": self.nuclei_list, "has_bg": self.has_background})
		except Exception as err:
			err.message = "Could not write input file for Matlab at %s" % path_to_mat_input
			raise err

		os.system("matlab -nodesktop -nosplash -r \"cd %s; run_seeded_watershed('%s', '%s')\"" % (matlab_m_file_path, path_to_mat_input, path_to_mat_result))
		os.system("stty echo")

		os.system('cp %s temp_workspace/init_watershed_all_time_stamps/init_ws.mat' % path_to_mat_result)


	

	def prepare_cellness_training_data(self):

		""" Copy the training data for the cellness_metric type selected by the user.
		"""

		if self.cellness_metric.lower() == "ascidian":

			# prepare training data:
			try:
				import CellECT
				data_location = os.system('cp %s/data/training/ascidian/* temp_workspace/training_data/' % CellECT.__path__[0])
			except err:
				err.message = "Could not copy cellness_metric training data."
				raise err



	def save_input_data(self):
		""" Save input data to mat: input vol, seeds 
		"""

		io.savemat("temp_workspace/init_watershed_all_time_stamps/input_ws.mat", {"vol": self.input_image, "seeds":self.nuclei_list})

		pass


	def write_to_config_file(self):

		""" Prepare the configuration file.
		"""

		try:
			self.conf_file = open("temp_workspace/config_files/config_file.cfg",'w')
		except err:
			err.message = "Could not create config file."
			raise err

		# write to config file
		try:

			self.conf_file.write('volume_mat_path = init_watershed_all_time_stamps/input_ws.mat\n')
			self.conf_file.write('volume_mat_var = vol\n')
			self.conf_file.write('first_seg_mat_path =  init_watershed_all_time_stamps/init_ws.mat\n')
			self.conf_file.write('first_seg_mat_var = ws\n')
			self.conf_file.write('nuclei_mat_path =  init_watershed_all_time_stamps/input_ws.mat\n')
			self.conf_file.write('nuclei_mat_var = seeds\n')
			self.conf_file.write('training_vol_mat_path =  training_data/positive_example.mat\n')
			self.conf_file.write('training_vol_mat_var = vol\n')
			self.conf_file.write('training_vol_nuclei_mat_path = training_data/positive_example.mat\n')
			self.conf_file.write('training_vol_nuclei_mat_var = seeds\n')
			self.conf_file.write('training_positive_seg_mat_path = training_data/positive_example.mat\n')
			self.conf_file.write('training_positive_seg_mat_var = label_map\n')
			self.conf_file.write('training_positive_labels_mat_path = training_data/positive_example.mat\n')
			self.conf_file.write('training_positive_labels_mat_var = labels\n')
			self.conf_file.write('training_negative_seg_mat_path = training_data/negative_example.mat\n')
			self.conf_file.write('training_negative_seg_mat_var = L\n')
			self.conf_file.write('training_negative_labels_mat_path = training_data/negative_example.mat\n')
			self.conf_file.write('training_negative_labels_mat_var = labels\n')
			self.conf_file.write('save_location_prefix = segs_all_time_stamps/timestamp_0_\n')
			self.conf_file.write('has_bg = 1\n')
			self.conf_file.write('use_size = 1\n')
			self.conf_file.write('use_border_intensity = 1\n')
			self.conf_file.write('use_border_distance = 0\n')
			self.conf_file.write('bq_token = %s\n' % self.access_token)
			self.conf_file.write('bq_mex_url = %s\n' % self.mex_url)
			self.conf_file.write('bq_t = %d\n' % self.time_stamp)
	
			self.conf_file.close()


		except err:
			err.message = "Could not write to config file."
			raise err




	def run(self):


#		self.mex_url = "http://localhost:8080/module_service/mex/53236"
#		self.access_token = "53236"


		self.setup_input()
		self.create_workspace("temp_workspace/")
		self.bqsession.update_mex("5% - Created workspace.")
		self.get_image()
		self.bqsession.update_mex("10% - Fetched image.")		
		self.get_ND3D_output()
		self.bqsession.update_mex("15% - Fetched ND3D output.")
		self.run_first_watershed()
		self.bqsession.update_mex("30% - Ran first watershed.")
		self.prepare_cellness_training_data()
		self.save_input_data()
		self.write_to_config_file()
		self.bqsession.update_mex("35% - Running CellECT.")
		self.run_CellECT()
		self.bqsession.update_mex("90% - Finished CellECT. Printing output.")
		self.attach_results()
		self.bqsession.finish_mex()




	def run_CellECT(self):

		# need to call CellECT_seg_tool script with python to preserve bisque env.

		import re
		output = os.popen('whereis CellECT_seg_tool').read()
		path_to_script = re.findall("CellECT_seg_tool:[\s](.*)CellECT_seg_tool[\s]", output)[0]

		os.system( 'python %s/CellECT_seg_tool -b -f temp_workspace/config_files/config_file.cfg ' % path_to_script )
	



	def start(self, args):

		# prepare session
		try:
			self.mex_url = args[1]
			self.access_token = args[2]
			self.run()

		except Exception, e:
			logging.exception ("Exception at %s" % e )
			self.bqsession.fail_mex(msg = "Exception at %s" % e)
			sys.exit(1)
		sys.exit(0)



if __name__ == "__main__":


	CellECT_SegTool_Wrapper().start(sys.argv)


