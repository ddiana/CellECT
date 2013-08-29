# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from libtiff import TIFF
from PySide import QtGui
import os
import os.path
import pdb


# Imports from this project
from CellECT.gui import newWorkspaceGui

from CellECT.workspace_management import metadata as md
from CellECT.workspace_management import workspace_creator
from CellECT.workspace_management import workspace_data
from CellECT.gui import meta_manager



class NewWorkspaceWindow(QtGui.QDialog, newWorkspaceGui.Ui_Dialog):

	def init_data(self):



		self.nuclei_csv = ""	

		self.last_dir = os.curdir

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


		self.meta_manager = meta_manager. ManageMetadataInUI(self, self.metadata)




	def import_nuclei_csv(self):

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")
		
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


		try:
			if self.ws_dir != None:
				self.ws_dir = self.last_dir
		except:
			self.ws_dir = self.last_dir

		if not len(self.nuclei_csv):
			put_random_nuclei = self.ask_if_random_nuclei()

		if not put_random_nuclei:
			# if the user wants to add nuclear detector output get back to the UI,
			# if not, continue to make the ws with random inputs.
			return

		self.ws_location = self.ws_dir + "/" + self.lineEdit_ws_name.text()
		self.new_ws = workspace_creator.WorkspaceCreator()
		self.new_ws.set_info(self.nuclei_csv, self.image_location, self.metadata, self.checkBox_has_bg.isChecked())
		self.new_ws.build_workspace(self.ws_location, self.progressBar )


		ws_obj = workspace_data.WorkSpaceData()
		ws_obj.metadata = self.metadata
		ws_obj.workspace_location = self.ws_location

		ws_obj.write_xml()

		
		self.main_ui.open_cws_file("%s/workspace_data.cws" % self.ws_location)

		self.close()

		

	def get_ws_location(self):

		dirname = QtGui.QFileDialog.getExistingDirectory(self, 'Open file', self.last_dir, options= QtGui.QFileDialog.ShowDirsOnly )

		self.last_dir = dirname
		self.ws_dir = dirname


	def load_metadata_from_csv(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")

		self.last_dir = os.path.dirname(filename)
		self.metadata.load_bq_csv_file(filename)
		self.metadata.populate_metadata_boxes(self)

	def load_metadata_from_xml(self):

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.xml);; All files (*.*)")

		self.last_dir = os.path.dirname(filename)
		self.metadata.load_bq_xml_file(filename)
		self.metadata.populate_metadata_boxes(self)

	def load_metadata_from_tif_info(self):

		self.metadata.load_info_from_tif(self.image_location)

		self.metadata.populate_metadata_boxes(self)	

	def load_stack(self):

		self.init_data()

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "TIFF (*.tif *.tiff);; All files (*.*)")
		

		self.last_dir = os.path.dirname(filename)
		self.image_location = filename

		tif = TIFF.open(self.image_location)

#		pic = tif.read_image()

		self.load_metadata_from_tif_info()
		pic = QtGui.QImage(filename)
		self.label_preview_img.setPixmap(QtGui.QPixmap.fromImage(pic))



