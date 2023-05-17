from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import random
import config
import torchvision.transforms.functional as TF

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        
        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "masks")
        
        
        # read images
        image_file_names = []
        for files in os.listdir(self.img_dir):
            image_file_names.append(os.path.join(self.img_dir, files))
        self.images = sorted(image_file_names)
        
        
        # read annotations
        annotation_file_names = []
        for files in os.listdir(self.ann_dir):
            annotation_file_names.append(os.path.join(self.ann_dir, files))
        self.annotations = sorted(annotation_file_names)
        
        
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def my_transforms(self, image, mask):

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image =  Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map =  Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        if self.train is True and config.ENABLE_AUG is True:
            # print("hello")
            image, segmentation_map = self.my_transforms(image, segmentation_map)
        
        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs