from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url

id2label = {0:'background', 1:'lane'}
label2id = {v: k for k, v in id2label.items()}

def build_segformer_feature_extractor():

	feature_extractor = SegformerFeatureExtractor(reduce_labels=False)

	return feature_extractor

def build_segformer_model():

	seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
		                                                         num_labels=2, 
		                                                         id2label=id2label, 
		                                                         label2id=label2id
		                                                         )

	
	return seg_model