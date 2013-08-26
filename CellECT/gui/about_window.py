from PySide import QtGui

from CellECT.gui import aboutGui

class AboutWindow(QtGui.QDialog, aboutGui.Ui_Dialog):

	def __init__(self, parent=None):
		super(AboutWindow, self).__init__(parent)
		
		self.setupUi(self)

		self.pushButton.clicked.connect(self.close)



