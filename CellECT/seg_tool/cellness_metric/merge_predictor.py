import heapq


class MergePredictor(object):

	def __init__(self, seg_collection):

		self.build_heap(seg_collection)


	def build_heap(self, seg_collection):

		self.to_merge_queue = []

		for segment in seg_collection.list_of_segments:
			self.to_merge_queue.extend(self.get_scores_with_neighbors(segment))

		heapq.heapify(self.to_merge_queue)



	def get_scores_with_neighbors(self, segment):

		scores = []

		for (label, score) in segment.feature_dict["weighted_merge_score"]:
			if label > segment.label:
				scores.append((score, label, segment.label))

		return scores

	def next_merge(self):
		
		return heapq.heappop(self.to_merge_queue)

