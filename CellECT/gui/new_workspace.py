# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from libtiff import TIFF
from PySide import QtGui
import os


# Imports from this project
from CellECT.gui import newWorkspaceGui

from CellECT.workspace_management import metadata as md
from CellECT.workspace_management import workspace_creator

class NewWorkspaceWindow(QtGui.QDialog, newWorkspaceGui.Ui_Dialog):

	
	def __init__(self, parent=None):
		super(NewWorkspaceWindow, self).__init__(parent)
		
		self.setupUi(self)

		self.btn_cancel.clicked.connect(self.close)

		self.btn_select_img.clicked.connect(self.load_stack)

		self.metadata = md.Metadata()
	
		self.btn_load_metadata.clicked.connect(self.load_metadata)

		self.btn_choose_location.clicked.connect(self.get_ws_location)

		self.btn_create_ws.clicked(self.create_workspace)


	def create_workspace(self):

		# TODO
		# check if ws name is given
		# check inputs


		self.ws_location = self.ws_dir + "/" + self.lineEdit_ws_name.text()
		self.new_ws = workspace_creator.WorkspaceCreator()
		self.new_ws.set_info(self.nuclei_csv, self.image_location, self.metadata)
		self.new_ws.build_workspace(self.ws_location)

		

	def get_ws_location(self):



		dirname, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', os.curdir,QtGui.QFileDialog.ShowDirsOnly )

		self.ws_dir = dirname


	def load_metadata(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', os.curdir, "Nuclei Detector Output (*.csv);; All files (*.*)")

		self.metadata.load_bq_csv_file(filename)
		self.metadata.populate_metadata_boxes(self)


	def load_stack(self):


		filename, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', os.curdir, "TIFF (*.tif *.tiff);; All files (*.*)")
		
		self.image_location = filename

		tif = TIFF.open(self.image_location)

#		pic = tif.read_image()

		pic = QtGui.QImage(filename)
		self.label_preview_img.setPixmap(QtGui.QPixmap.fromImage(pic))
