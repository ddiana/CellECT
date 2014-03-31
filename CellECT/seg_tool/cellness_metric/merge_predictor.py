import heapq
import pdb


from CellECT.seg_tool.features.segment_features import get_bounding_box_union_two_segments
from CellECT.seg_tool.gui.predict_merge_gui import MergePredictorUI


class MergePredictor(object):

	def __init__(self, seg_collection, vol, vol_nuclei, label_map, color_map=None):


		self.seg_collection = seg_collection
		self.vol = vol
		self.vol_nuclei = vol_nuclei
		self.build_heap()
		self.label_map = label_map
		self.ui = None
		self.list_to_merge = []
		self.color_map = color_map
		self.all_answers = []



	def build_heap(self):

		self.to_merge_queue = []

		for segment in self.seg_collection.list_of_segments:
			self.to_merge_queue.extend(self.get_scores_with_neighbors(segment))

		heapq.heapify(self.to_merge_queue)



	def get_scores_with_neighbors(self, segment):

		scores = []

		try:
			for (label, score) in segment.feature_dict["weighted_merge_score"]:
				if label > segment.label:
					scores.append((score, label, segment.label))
		except:
			pdb.set_trace()


		return scores

	def next_merge(self):
		
		if len (self.to_merge_queue):

			merge_tuple =  heapq.heappop(self.to_merge_queue)

			idx1 = self.seg_collection.segment_label_to_list_index_dict[merge_tuple[1]]
			idx2 = self.seg_collection.segment_label_to_list_index_dict[merge_tuple[2]]
		
			seg1 = self.seg_collection.list_of_segments[idx1]
			seg2 = self.seg_collection.list_of_segments[idx2]

			bbx = get_bounding_box_union_two_segments(seg1, seg2)

			vol = self.vol[bbx.xmin:bbx.xmax,  bbx.ymin:bbx.ymax,  bbx.zmin:bbx.zmax]
			vol_nuclei = self.vol_nuclei[bbx.xmin:bbx.xmax,  bbx.ymin:bbx.ymax,  bbx.zmin:bbx.zmax]
			label_map = self.label_map[bbx.xmin:bbx.xmax,  bbx.ymin:bbx.ymax,  bbx.zmin:bbx.zmax]
			highlight_map = (label_map == 0) + 100*(label_map == seg1.label) + 200*(label_map == seg2.label)

			self.ui = MergePredictorUI()
			self.ui.set_data(vol, vol_nuclei, label_map, highlight_map, self.color_map, seg1.label, seg2.label, self.list_to_merge, self.all_answers, merge_tuple[0])
			self.ui.display()

		else:

			print "Nothing to merge."

		




