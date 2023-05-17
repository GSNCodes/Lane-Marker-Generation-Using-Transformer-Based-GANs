import warnings
warnings.filterwarnings("ignore")
import torch
import config
import argparse
import numpy as np
from torch import nn
from torchmetrics import Dice
import matplotlib.pyplot as plt
from datasets import load_metric
from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset
from segformer_model import build_segformer_feature_extractor, id2label, label2id
from torchmetrics.image.fid import FrechetInceptionDistance


def fid_calc(test_dataloader, model, device):
    model.eval()

    fid = FrechetInceptionDistance(feature=64)
    
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            labels = labels[:,None, :, :]
            predicted = predicted[:,None, :, :]

            labels = torch.cat([labels, labels, labels], dim=1)
            predicted = torch.cat([predicted, predicted, predicted], dim=1)


            # print(labels.shape)
            fid.update(labels.detach().cpu().type(torch.uint8), real=True)
            fid.update(predicted.detach().cpu().type(torch.uint8), real=False)


    print("FID Score:", fid.compute())


def perform_evaluation(test_dataloader, model, device):
    
    metric = load_metric("mean_iou")
    f1_metric = load_metric("f1")
    p_metric = load_metric("precision")
    r_metric = load_metric("recall")

    
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            
            metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
            # f1_metric.add_batch(predictions=predicted.detach().cpu().numpy().flatten(), references=labels.detach().cpu().numpy().flatten())
            # p_metric.add_batch(predictions=predicted.detach().cpu().numpy().flatten(), references=labels.detach().cpu().numpy().flatten())
            # r_metric.add_batch(predictions=predicted.detach().cpu().numpy().flatten(), references=labels.detach().cpu().numpy().flatten())

    metrics = metric.compute(num_labels=len(id2label), 
                             ignore_index=255,
                             reduce_labels=False, # we've already reduced the labels before)
                            )
    # f1_metrics = f1_metric.compute()
    # p_metrics = p_metric.compute()
    # r_metrics = r_metric.compute()

    print("[LOG] ---- Printing Stats")
    print("Loss:", outputs.loss.item())
    print("Mean_iou:", metrics["mean_iou"])
    # print("F1-Score: ", f1_metrics["f1"])
    # print("Precison: ", p_metrics["precision"])
    # print("Recall: ", r_metrics["recall"])
    print("Mean accuracy:", metrics["mean_accuracy"])

def visualize(test_dataloader, model, device):
    model.eval()

    idx = int(torch.randint(0, len(test_dataloader), (1,))[0])

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            img1 = labels.detach().cpu().numpy()[0].reshape((512, 512, 1))
            img2 = predicted.detach().cpu().numpy()[0].reshape((512, 512, 1))

            print(img1.shape, img2.shape)

            import cv2

            temp = np.concatenate((img1, img2), axis=1)

            print(temp.shape)

            plt.imshow(temp)

            plt.show()

            break



if __name__ == "__main__":

    #############################################################
    ######################## Arguments ##########################

    parser = argparse.ArgumentParser(
                        prog='SowmiyaCapstone-Test',
                        description='Generate Test Metrics'
                        )

    parser.add_argument('-w', '--weights', required=True, type=str, help="Weights of the trained model")

    args = parser.parse_args()
    print(f" ---- [INFO] Using weights file: {args.weights} ----")

    #############################################################
    #############################################################

    print(" ---- [INFO] TESTING IN PROGRESS ........")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = build_segformer_feature_extractor()

    test_root_dir = config.TEST_ROOT_DIR
    test_dataset = SemanticSegmentationDataset(root_dir=test_root_dir, feature_extractor=feature_extractor, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    test_model = torch.load(args.weights).to(device)
    perform_evaluation(test_dataloader, test_model, device)
    fid_calc(test_dataloader, test_model, device)