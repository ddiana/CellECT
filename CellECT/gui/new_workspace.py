# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from libtiff import TIFF
from PySide import QtGui
import os
import os.path
import pdb
from scipy import misc

# Imports from this project
from CellECT.gui import newWorkspaceGui
from CellECT.workspace_management import metadata as md
from CellECT.workspace_management import workspace_creator
from CellECT.workspace_management import workspace_data
from CellECT.gui import meta_manager
from CellECT.gui import nuclei_options



class NewWorkspaceWindow(QtGui.QDialog, newWorkspaceGui.Ui_Dialog):

	def init_data(self):



		self.nuclei_csv = ""	

		self.last_dir = os.curdir
		self.image_location = None

		self.metadata = md.Metadata()
	

	def __init__(self, parent=None):

		super(NewWorkspaceWindow, self).__init__(parent)

		self.main_ui = parent	
		self.setupUi(self)

		self.init_data()

		self.btn_cancel.clicked.connect(self.close)

		self.btn_select_img.clicked.connect(self.load_stack)

		self.btn_load_metadata_csv.clicked.connect(self.load_metadata_from_csv)
		self.btn_load_metadata_xml.clicked.connect(self.load_metadata_from_xml)

		self.btn_choose_location.clicked.connect(self.get_ws_location)

		self.btn_create_ws.clicked.connect(self.create_workspace)

		self.btn_import_nuclei_csv.clicked.connect(self.import_nuclei_csv)

		self.meta_manager = meta_manager.ManageMetadataInUI(self, self.metadata)




	def import_nuclei_csv(self):

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")

		if not len(filename):
			return
	
		self.last_dir = os.path.dirname(filename)

		self.nuclei_csv = filename



	def ask_if_random_nuclei(self):

		flags = QtGui.QMessageBox.StandardButton.Yes 
		flags |= QtGui.QMessageBox.StandardButton.No

		question = "No nuclear detector output selected. Place random seeds?"

		response = QtGui.QMessageBox.question(self, "Question", question, flags)


		if response == QtGui.QMessageBox.Yes:
			return True
		elif QtGui.QMessageBox.No:
			return False
	
	


	def create_workspace(self):

		# TODO
		# check if ws name is given
		# check inputs

		self.metadata = self.meta_manager.metadata


		if self.image_location == None:
			
			QtGui.QMessageBox.information(self, "CellECT New Workspace", "No input image selected. Cannot create workspace.")
			return

		if self.metadata.num_pages != self.metadata.numz * self.metadata.numt * self.metadata.numch:
			message = "Metadata information incorrect. There are %d pages in this image stack. Metadata must satisfy T*Z*C = %d. Cannot create workspace." % (self.metadata.num_pages, self.metadata.num_pages)
			QtGui.QMessageBox.information(self, "CellECT New Workspace", message )
			return	

		if self.metadata.numx == 0 or self.metadata.numy == 0 or self.metadata.numch ==0:

			QtGui.QMessageBox.information(self, "CellECT New Workspace", "Metadata information incomplete. Cannot create workspace.")
			return

		if self.lineEdit_ws_name.text() == "":
			QtGui.QMessageBox.information(self, "CellECT New Workspace", "Please select a workspace name.")
			return

		if self.metadata.numz <3 :

			QtGui.QMessageBox.information(self, "CellECT New Workspace", "Input image must be 3D or 4D. If metadata is incorrect, please modify it.")
			return


		try:
			if self.ws_dir != None:
				self.ws_dir = self.last_dir
		except:
			self.ws_dir = self.last_dir

		action = "do_nothing"

		if not len(self.nuclei_csv):
			dlg = nuclei_options.NucleiOptionsGui()
			dlg.exec_()
			action = dlg.action
		else:
			action = "has_nuclei"



		if action == "do_nothing":
			# if the user wants to add nuclear detector output get back to the UI,
			# if not, continue to make the ws with random inputs.
			return




		self.ws_location = os.path.join(self.ws_dir, self.lineEdit_ws_name.text())
		self.new_ws = workspace_creator.WorkspaceCreator()

		self.new_ws.set_info(self.nuclei_csv, self.image_location, self.metadata, False, action)

	
		try:
			self.new_ws.build_workspace(self.ws_location, self.progressBar )
		except IOError as err:
			QtGui.QMessageBox.information(self, "CellECT New Workspace", err.message)			
		except Exception as err:
			QtGui.QMessageBox.information(self, "CellECT New Workspace", "Could not create workspace. Error: %s" % err)
			return
		



		ws_obj = workspace_data.WorkSpaceData()
		ws_obj.metadata = self.metadata
		ws_obj.workspace_location = self.ws_location

		ws_obj.write_xml()

		
		self.main_ui.open_cws_file(os.path.join(self.ws_location,"workspace_data.cws"))

		self.close()

		

	def get_ws_location(self):

		dirname = QtGui.QFileDialog.getExistingDirectory(self, 'Open file', self.last_dir, options= QtGui.QFileDialog.ShowDirsOnly )

		if not len(dirname):
			return 

		self.last_dir = dirname
		self.ws_dir = dirname


	def load_metadata_from_csv(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")

		if not len(filename):
			return 

		self.last_dir = os.path.dirname(filename)
		self.metadata.load_bq_csv_file(filename)
		self.metadata.populate_metadata_boxes(self)

	def load_metadata_from_xml(self):

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.xml);; All files (*.*)")

		if not len(filename):
			return

		self.last_dir = os.path.dirname(filename)
		self.metadata.load_bq_xml_file(filename)
		self.metadata.populate_metadata_boxes(self)

	def load_metadata_from_tif_info(self):

		self.metadata.load_info_from_tif(self.image_location)

		self.meta_manager.metadata = self.metadata
		# TODO: fix this double variable shit!!!!!!!!
		self.metadata.populate_metadata_boxes(self)	

	def load_stack(self):



		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "TIFF (*.tif *.tiff);; All files (*.*)")
		
		if not len(filename):
			return

		self.init_data()
		self.last_dir = os.path.dirname(filename)
		self.image_location = filename
		

		tif = TIFF.open(self.image_location)

		pic = tif.read_image()
		misc.imsave("temp.jpg", pic)
	
		# TODO: wait message
		#msg_box = QtGui.QMessageBox.information(self, "CellECT New Workspace", "Loading a large stack may take time. Press OK to continue.", defaultB = QtGui.QMessageBox.NoButton )
		self.load_metadata_from_tif_info()

		#pic = QtGui.QImage(filename)
		#self.label_preview_img.setPixmap(QtGui.QPixmap.fromImage(pic))
		self.label_preview_img.setPixmap("temp.jpg")


