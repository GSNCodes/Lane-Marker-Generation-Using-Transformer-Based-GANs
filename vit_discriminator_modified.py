from transformers import ViTImageProcessor
from transformers import ViTForImageClassification

id2label = {0:'fake', 1:'real'}
label2id = {v: k for k, v in id2label.items()}

def build_vit_image_processor():

	processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

	return processor

def build_vit_model():

	vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=2,
                                                              id2label=id2label,
                                                              label2id=label2id)

	new_config = vit_model.config

	new_config.num_channels = 4

	new_model = ViTForImageClassification(new_config)

	vit_model.vit.embeddings = new_model.vit.embeddings

	return vit_model