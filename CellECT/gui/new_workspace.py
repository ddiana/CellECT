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

class NewWorkspaceWindow(QtGui.QDialog, newWorkspaceGui.Ui_Dialog):

	
	def __init__(self, parent=None):
		super(NewWorkspaceWindow, self).__init__(parent)

		self.main_ui = parent		

		self.last_dir = os.curdir
		self.setupUi(self)

		self.btn_cancel.clicked.connect(self.close)

		self.btn_select_img.clicked.connect(self.load_stack)

		self.metadata = md.Metadata()
	
		self.btn_load_metadata.clicked.connect(self.load_metadata)

		self.btn_choose_location.clicked.connect(self.get_ws_location)

		self.btn_create_ws.clicked.connect(self.create_workspace)

		self.btn_import_nuclei_csv.clicked.connect(self.import_nuclei_csv)



	def import_nuclei_csv(self):

		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")
		
		self.last_dir = os.path.dirname(filename)

		self.nuclei_csv = filename


	def create_workspace(self):

		# TODO
		# check if ws name is given
		# check inputs


		try:
			if self.ws_dir != None:
				self.ws_dir = self.last_dir
		except:
			self.ws_dir = self.last_dir

		self.ws_location = self.ws_dir + "/" + self.lineEdit_ws_name.text()
		self.new_ws = workspace_creator.WorkspaceCreator()
		self.new_ws.set_info(self.nuclei_csv, self.image_location, self.metadata)
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


	def load_metadata(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "Nuclei Detector Output (*.csv);; All files (*.*)")

		self.last_dir = os.path.dirname(filename)
		self.metadata.load_bq_csv_file(filename)
		self.metadata.populate_metadata_boxes(self)


	def load_stack(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.last_dir, "TIFF (*.tif *.tiff);; All files (*.*)")
		

		self.last_dir = os.path.dirname(filename)
		self.image_location = filename

		tif = TIFF.open(self.image_location)

#		pic = tif.read_image()

		pic = QtGui.QImage(filename)
		self.label_preview_img.setPixmap(QtGui.QPixmap.fromImage(pic))



