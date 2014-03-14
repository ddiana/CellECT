import heapq


from CellECT.seg_tool.features.segment_features import get_bounding_box_union_two_segments
from CellECT.seg_tool.gui.predict_merge_gui import MergePredictorUI


class MergePredictor(object):

	def __init__(self, seg_collection, vol, label_map):

		self.build_heap(seg_collection)
		self.seg_collection = seg_collection
		self.label_map = label_map
		self.ui = None
		self.user_feedback = None
		self.color_map = None


	def build_heap(self):

		self.to_merge_queue = []

		for segment in self.seg_collection.list_of_segments:
			self.to_merge_queue.extend(self.get_scores_with_neighbors(segment))

		heapq.heapify(self.to_merge_queue)



	def get_scores_with_neighbors(self, segment):

		scores = []

		for (label, score) in segment.feature_dict["weighted_merge_score"]:
			if label > segment.label:
				scores.append((score, label, segment.label))

		return scores

	def next_merge(self):
		
		merge_tuple =  heapq.heappop(self.to_merge_queue)

		idx1 = self.seg_collection.segment_label_to_list_index_dict[merge_tuple[1]]
		idx2 = self.seg_collection.segment_label_to_list_index_dict[merge_tuple[2]]
		
		seg1 = self.seg_collection.list_of_segments[idx1]
		seg2 = self.seg_collection.list_of_segments[ids2]

		bbx = get_bounding_box_union_two_segments(seg1.bounding_box, seg2.bounding_box)

		vol = self.vol[bbx.xmin:bbx.xmax,  bbx.ymin:bbx.ymax,  bbx.zmin:bbx.zmax]
		label_map = self.vol[bbx.xmin:bbx.xmax,  bbx.ymin:bbx.ymax,  bbx.zmin:bbx.zmax]
		highlight_map = (label_map == 0) + 100*(label_map == seg1.label) + 200*(label_map == seg.label2)

		self.ui = MergePredictorUI()
		self.ui.set_data(vol, label_map, highlight_map)
		self.ui.display()
		self.user_feedback = self.ui.user_feedback
		
		




