# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

# Imports
from PySide import QtGui
from PySide import QtCore

# Imports from this project
from CellECT.gui import nucleiOptionsGui




class NucleiOptionsGui(QtGui.QDialog, nucleiOptionsGui.Ui_Dialog):

	def __init__(self, parent=None):

		super(NucleiOptionsGui, self).__init__(parent)
		self.setupUi(self)

		self.btn_cancel.clicked.connect(self.go_back_do_nothing)

		self.radioButton_no_nuclei.clicked.connect(self.set_no_nuclei)
		#self.radioButton_estimate.clicked.connect(self.set_estimate)
		self.radioButton_random.clicked.connect(self.set_random)
		self.btn_continue.clicked.connect(self.close)
		self.action = "no_nuclei"


	def getValues(self):
		return self.action


	def set_no_nuclei(self):

		self.action = "no_nuclei"

	def set_estimate(self):
		self.action = "use_estimate"

	def set_random(self):
		self.action = "use_random"


	def go_back_do_nothing(self):

		self.action = "do_nothing"

		self.close()	
