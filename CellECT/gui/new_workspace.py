# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports

from PySide import QtGui

# Imports from this project
from CellECT.gui import newWorkspaceGui



class NewWorkspaceWindow(QtGui.QDialog, newWorkspaceGui.Ui_Dialog):

	
	def __init__(self, parent=None):
		super(NewWorkspaceWindow, self).__init__(parent)
		
		self.setupUi(self)

		self.btn_cancel.clicked.connect(self.close)
