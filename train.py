from torch.utils.data import DataLoader
from dataset import SemanticSegmentationDataset
from segformer_model import build_segformer_feature_extractor, build_segformer_model, id2label, label2id
from vit_discriminator_modified import build_vit_image_processor, build_vit_model
import torch
import config
from dataset import SemanticSegmentationDataset
from segformer_model import build_segformer_feature_extractor
from torch import nn
from datasets import load_metric
import argparse
from tqdm import tqdm
import torchvision
from functools import reduce
import os
from utils import get_module_by_name, get_activation, activation, log_hyperparams


#############################################################
################## Arguments ################################
parser = argparse.ArgumentParser(
                    prog='SowmiyaCapstone',
                    description='Transformer based GAN for Lane Segmentation'
                    )

parser.add_argument('-e', '--exp_name', required=True, type=str, help="Name of the experiment")

args = parser.parse_args()
print(f" ---- [INFO] Experiment Name: {args.exp_name} ----")

#############################################################
################## Generator ################################

feature_extractor = build_segformer_feature_extractor()
generator_model = build_segformer_model()

#############################################################
################## Discriminator ############################

processor = build_vit_image_processor()
image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]

discriminator_model = build_vit_model()

#############################################################
#############################################################
train_root_dir = config.TRAIN_ROOT_DIR
test_root_dir = config.TEST_ROOT_DIR

train_dataset = SemanticSegmentationDataset(root_dir=train_root_dir, feature_extractor=feature_extractor)
test_dataset = SemanticSegmentationDataset(root_dir=test_root_dir, feature_extractor=feature_extractor)

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

#############################################################
#############################################################

# define optimizer
generator_optimizer = torch.optim.AdamW(generator_model.parameters(), lr=config.GR_LEARNING_RATE)
discriminator_optimizer = torch.optim.AdamW(generator_model.parameters(), lr=config.DR_LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=100, gamma=0.1)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---- [INFO] Device Used: ---- {device}")
generator_model.to(device)
discriminator_model.to(device)
# Create metric for evaluation
generator_metric = load_metric("mean_iou")
discriminator_metric = load_metric("accuracy")
# define criterion
criterion = nn.CrossEntropyLoss()

if config.EMBEDDING_LOSS_TYPE == "cosine":
    embedding_criterion = nn.CosineSimilarity(dim=-1)
else if config.EMBEDDING_LOSS_TYPE == "l2":
    embedding_criterion = nn.MSELoss()
else:
    raise("Unknown loss chosen. Check whether the EMBEDDING_LOSS_TYPE parameter is set to either of the following [\"cosine\", \"l2\"] ")
#############################################################
#############################################################

train_discriminator = True
torch.backends.cudnn.benchmark = True

#############################################################
#############################################################

if os.path.isdir("logs") is False:
    os.mkdir("logs")

loss_file = os.path.join("logs", args.exp_name + "_loss.txt")
loss_fh = open(loss_file, 'w+')

#############################################################
#############################################################

log_hyperparams(args.exp_name)
best_loss = None
best_accuracy = None
generator_weights_path = os.path.join("weights", args.exp_name + "_generator_weights.pt")
best_generator_weights_path = os.path.join("weights", args.exp_name + "_generator_weights_best.pt")
#############################################################
#############################################################


for epoch in range(config.NUM_EPOCHS):  # loop over the dataset multiple times
    
    print("Epoch:", epoch)
    
    for idx, batch in enumerate(tqdm(train_dataloader)):
        
        # Get the Inputs
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)


        labels_real = torch.zeros((labels.shape[0],2)).to(device)
        labels_real[:,1] = 1
        labels_fake = torch.zeros((labels.shape[0],2)).to(device)
        labels_fake[:,0] = 1

        # print(labels_real)
        # print(labels_fake)

        #############

        if train_discriminator and epoch > config.GR_PRE_TRAINING_EPOCHS:
            discriminator_model.train()
            generator_model.eval()

            discriminator_optimizer.zero_grad()

            ######## Train using GroundTruth Masks ##########################
            discriminator_labels = labels[:,None, :, :]

            discriminator_labels = torch.cat((pixel_values, discriminator_labels), dim=1)


            discrimator_label_values = torchvision.transforms.Resize((224, 224))(discriminator_labels)
            # print(discrimator_label_values.shape)

            discriminator_model.vit.layernorm.register_forward_hook(get_activation('d1'))
            test_output = discriminator_model(pixel_values=discrimator_label_values)

            temp1 = activation['d1']

            loss_d_real = criterion(test_output.logits, labels_real)
            loss_d_real.backward()
            
            ######## Train using Predicted Masks ##########################

            discriminator_labels = generator_model(pixel_values=pixel_values, labels=labels).logits
            discriminator_labels = discriminator_labels.argmax(dim=1)[:,None, :, :]
    
            discrimator_label_values = torchvision.transforms.Resize((224, 224))(discriminator_labels)
            pixel_label_values = torchvision.transforms.Resize((224, 224))(pixel_values)
            discriminator_labels_final = torch.cat((pixel_label_values, discrimator_label_values), dim=1)
            test_output = discriminator_model(pixel_values=discriminator_labels_final)

            loss_d_fake = criterion(test_output.logits, labels_fake)
            loss_d_fake.backward()

            discriminator_optimizer.step()

        #################

        generator_model.train()

        if epoch > config.GR_PRE_TRAINING_EPOCHS:
            discriminator_model.eval()

        # zero the parameter gradients
        generator_optimizer.zero_grad()

        # forward + backward + optimize
        generator_output = generator_model(pixel_values=pixel_values, labels=labels)
        generator_loss, generator_logits = generator_output.loss, generator_output.logits

        if epoch > config.GR_PRE_TRAINING_EPOCHS:

            generator_labels = generator_logits
            generator_labels = generator_labels.argmax(dim=1)[:,None, :, :]

            generator_label_values = torchvision.transforms.Resize((224, 224))(generator_labels)
            pixel_label_values = torchvision.transforms.Resize((224, 224))(pixel_values)
            discriminator_model.vit.layernorm.register_forward_hook(get_activation('d2'))

            generator_labels_final = torch.cat((pixel_label_values, generator_label_values), dim=1)
            output = discriminator_model(pixel_values=generator_labels_final)

            temp2 = activation['d2']

            loss_g1 = criterion(output.logits, labels_real)

            loss_g2 = embedding_criterion(temp1.detach(), temp2)
            
            total_generator_loss = loss_g1 + torch.mean(loss_g2) + generator_loss

            total_generator_loss.backward()

        else:
            generator_loss.backward()

        generator_optimizer.step()

        # evaluate
        with torch.no_grad():
            upsampled_logits = nn.functional.interpolate(generator_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)
            
            # note that the metric expects predictions + labels as numpy arrays
            generator_metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Print loss and metrics every 100 batches
        if (idx) % 100 == 0:
            metrics = generator_metric.compute(num_labels=len(id2label), 
                                   ignore_index=255,
                                   reduce_labels=False, # we've already reduced the labels before)
                                    )

            print("Loss:", generator_loss.item())
            print("Mean_iou:", metrics["mean_iou"])
            print("Mean accuracy:", metrics["mean_accuracy"])

            if best_accuracy is None:
                best_accuracy = metrics["mean_accuracy"]

            elif metrics["mean_accuracy"] > best_accuracy:
                best_accuracy = metrics["mean_accuracy"]
                torch.save(generator_model, best_generator_weights_path)

            # if best_loss is None:
            #     best_loss = generator_loss.item()
            
            # elif generator_loss.item() < best_loss:
            #     best_loss = generator_loss.item()
            #     torch.save(generator_model, best_generator_weights_path)

    scheduler.step()

    torch.save(generator_model, generator_weights_path)

    loss_fh.write(f"Epoch{epoch} " + str(generator_loss.item()) + '\n')

loss_fh.close()

torch.save(generator_model, generator_weights_path)

if config.SAVE_DISCRIMINATOR_WEIGHTS is True:
    discriminator_weights_path = os.path.join("weights", args.exp_name + "_discriminator_weights.pt")
    torch.save(discriminator_model.state_dict(), discriminator_weights_path)


