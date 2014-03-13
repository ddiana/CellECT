import pylab


class MergePredictorUI(object):

	def __init__(self,)

		self.vol = None
		self.label_map = None
		self.highlight_map = None

		self.answer = None


	def set_data(self, vol, label_map, highlight_map)

		self.vol = vol
		self.label_map = label_map
		self.highlight_map = highlight_map


	def display(self):

		self.fig = pylab.figure()

		


