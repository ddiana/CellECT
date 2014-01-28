# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from PySide import QtGui
from PySide import QtCore
import pdb

# Imports from this project
from CellECT.gui import displayProgress




class DisplayProgress(QtGui.QDialog, displayProgress.Ui_Dialog):


	def __init__(self, parent=None):

		super(DisplayProgress, self).__init__(parent)

		self.setupUi(self)
		self.val = 10

		

	def set_progress(self, val, text):


	
		self.progressBar.setValue(val)		
		self.label_status.setText(text)
