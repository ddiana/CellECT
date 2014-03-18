import pdb

from scipy import io


class SplitPredict(object):

	def __init__ (self, vol, label_map, segment_collection):


		self.choose_segments(segment_collection)
		self.prepare_segments(segment_collection, vol, label_map)
		self.process_segments()

		


	def choose_segments(self, segment_collection):

		# choose the segments with the highest score 

		max_size = max((seg.feature_dict["size"] for seg in segment_collection.list_of_segments))

		get_score = lambda seg, ms: 0.2*seg.feature_dict["interior_weighted_intensity_mean"] + 0.8* seg.feature_dict["size"]/float(ms)

		self.list_of_candidates = [(get_score(seg, max_size), seg.label) for seg in segment_collection.list_of_segments]
		
		self.list_of_candidates = sorted(self.list_of_candidates)
		
		if len(self.list_of_candidates) > 300:
			self.list_of_candidates = self.list_of_candidates[-300:]

		
	def prepare_segments(self, segment_collection, vol, label_map):

		segments_to_process = []

		for score, label in self.list_of_candidates:
			idx = segment_collection.segment_label_to_list_index_dict[label]
			seg = segment_collection.list_of_segments[idx]

			bbx = seg.bounding_box
			crop_vol = vol[bbx.xmin:bbx.xmax, bbx.ymin:bbx.ymax, bbx.zmin:bbx.zmax]
			crop_map = label_map[bbx.xmin:bbx.xmax, bbx.ymin:bbx.ymax, bbx.zmin:bbx.zmax]

			this_segment = {}
			this_segment["label"] = seg.label 
			this_segment["vol"] = crop_vol
			this_segment["map"] = crop_map	

			segments_to_process.append(this_segment)

		self.save_location = "test1.mat"
		
		io.savemat(self.save_location,{"segments": segments_to_process})

		pdb.set_trace()


	def process_segments(self):

		pass

