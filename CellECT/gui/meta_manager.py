# Author: Diana Delibaltov
# Vision Research Lab, University of California Santa Barbara

import pdb

class ManageMetadataInUI(object):


	def __init__(self, ui, metadata):

		self.ui = ui
		self.metadata = metadata

		self.ui.spinBox_numx.valueChanged.connect(self.update_metadata_numx)
		self.ui.spinBox_numy.valueChanged.connect(self.update_metadata_numy)
		self.ui.spinBox_numz.valueChanged.connect(self.update_metadata_numz)
		self.ui.spinBox_numt.valueChanged.connect(self.update_metadata_numt)
		self.ui.doubleSpinBox_xres.valueChanged.connect(self.update_metadata_xres)
		self.ui.doubleSpinBox_yres.valueChanged.connect(self.update_metadata_yres)
		self.ui.doubleSpinBox_zres.valueChanged.connect(self.update_metadata_zres)
		self.ui.doubleSpinBox_tres.valueChanged.connect(self.update_metadata_tres)
		self.ui.spinBox_numch.valueChanged.connect(self.make_list_of_channels_in_combobox)
		#self.ui.comboBox_mem_chan.currentIndexChanged.connect(self.update_membrane_channel)

	def update_metadata_xres(self):

		if self.ui.doubleSpinBox_xres.value():
			self.metadata.xres = self.ui.doubleSpinBox_xres.value()

	def update_metadata_yres(self):

		if self.ui.doubleSpinBox_yres.value():
			self.metadata.yres = self.ui.doubleSpinBox_yres.value()

	def update_metadata_zres(self):

		if self.ui.doubleSpinBox_zres.value():
			self.metadata.zres = self.ui.doubleSpinBox_zres.value()

	def update_metadata_tres(self):

		if self.ui.doubleSpinBox_tres .value():
			self.metadata.tres = self.ui.doubleSpinBox_tres .value()

	def update_metadata_numx(self):

		if self.ui.spinBox_numx.value():
			self.metadata.numx = self.ui.spinBox_numx.value()

	def update_metadata_numy(self):

		if self.ui.spinBox_numy .value():
			self.metadata.numy = self.ui.spinBox_numy .value()

	def update_metadata_numz(self):

		if self.ui.spinBox_numz.value():
			self.metadata.numz = self.ui.spinBox_numz.value()

	def update_metadata_numt(self):

		if self.ui.spinBox_numt.value():
			self.metadata.numt = self.ui.spinBox_numt.value()


#	def update_metadata_numch(self):

#		if self.ui.spinBox_numch.value():
#			self.metadata.numch = self.ui.spinBox_numch.value()

#		self.make_list_of_channels_in_combobox(self.ui.spinBox_numch.value())




	def make_list_of_channels_in_combobox(self, value):

		if self.ui.spinBox_numch.value():
			self.metadata.numch = self.ui.spinBox_numch.value()

		try:
			current_index = self.ui.comboBox_mem_chan.currentIndex()
		except:
			pass

		self.ui.comboBox_mem_chan.clear()
		self.ui.comboBox_mem_chan.addItems([str(i) for i in xrange(value)])
		self.ui.comboBox_mem_chan.setCurrentIndex(current_index)


	def update_membrane_channel(self):

		self.metadata.memch = self.ui.comboBox_mem_chan.currentIndex()



