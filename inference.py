import os
import cv2
import torch
import config
import argparse
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset
from segformer_model import build_segformer_feature_extractor, id2label, label2id


def perform_inference(feature_extractor, model, device, save_folder="results", input_dir=None):
    model.eval()

    if input_dir is None:
        input_dir = os.path.join(config.TEST_ROOT_DIR, "images")

    for filename in os.listdir(input_dir):

        img_filepath = os.path.join(input_dir, filename)

        input_img = cv2.imread(img_filepath)

        encoded_inputs = feature_extractor(images=input_img, return_tensors="pt")

        encoded_inputs = encoded_inputs.to(device)

        with torch.no_grad():

            outputs = model(**encoded_inputs)
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=input_img.shape[:2], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

        save_path = os.path.join(save_folder, filename.replace("jpg", "png"))

        cv2.imwrite(save_path, predicted[0].cpu().numpy()*255)




if __name__ == "__main__":

    #############################################################
    ######################## Arguments ##########################

    parser = argparse.ArgumentParser(
                        prog='SowmiyaCapstone-Inference',
                        description='Perform inference on images'
                        )

    parser.add_argument('-w', '--weights', required=True, type=str, help="Weights of the trained model")
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="Path to store output")
    parser.add_argument('-i', '--input_dir', required=False, type=str, default=None, help="Path to folder containing test images")

    args = parser.parse_args()
    print(f" ---- [INFO] Using weights file: {args.weights} ----")

    #############################################################
    #############################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = build_segformer_feature_extractor()

    test_model = torch.load(args.weights).to(device)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    perform_inference(feature_extractor, test_model, device, input_dir=args.input_dir, save_folder=args.output_dir)
