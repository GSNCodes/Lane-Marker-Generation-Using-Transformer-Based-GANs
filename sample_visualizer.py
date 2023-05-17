import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


debug = False

test_root_dir = '/home/gsn/Spring-2023-Finale/Capstone/GSNDGML/DatasetCapstone/Duluth_1000/test_set'

save_folder = os.path.join(test_root_dir, "result_compared_l2_100_dice")


if os.path.exists(save_folder) is False:
	os.makedirs(save_folder)


img_folder = os.path.join(test_root_dir, "images")
gt_mask_folder = os.path.join(test_root_dir, "masks")
pd_mask_folder = os.path.join(test_root_dir, "results_l2_100_dice")


num_samples = 100
my_color_palette = [[0,0,0], [0, 255, 0]]
img_grid = []
for idx, filename in enumerate(os.listdir(img_folder)):

	if idx >= num_samples:
		break

	print(filename)

	temp = []

	image_path = os.path.join(img_folder, filename)
	gt_mask_path = os.path.join(gt_mask_folder, filename.replace("jpg", "png"))
	pd_mask_path = os.path.join(pd_mask_folder, filename.replace("jpg", "png"))

	image = cv2.imread(image_path)
	image = cv2.resize(image, (512, 512))

	gt_mask = cv2.imread(gt_mask_path,0)
	gt_mask = cv2.resize(gt_mask, (512, 512))
	# gt_mask = cv2.cvtColor(gt_mask,cv2.COLOR_GRAY2RGB)

	pd_mask = cv2.imread(pd_mask_path,0) // 255
	# print(np.unique(pd_mask))
	# pd_mask = cv2.cvtColor(pd_mask,cv2.COLOR_GRAY2RGB)


	color_seg = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8) # height, width, 3
	palette = np.array(my_color_palette)
	for label, color in enumerate(palette):
	    color_seg[gt_mask == label, :] = color


	result1 = image * 0.9 + color_seg * 0.5
	result1 = result1.astype(np.uint8)

	color_seg = np.zeros((pd_mask.shape[0], pd_mask.shape[1], 3), dtype=np.uint8) # height, width, 3
	palette = np.array(my_color_palette)
	for label, color in enumerate(palette):
	    color_seg[pd_mask == label, :] = color

	result2 = image * 0.9 + color_seg * 0.5
	result2 = result2.astype(np.uint8)

	

	result = cv2.hconcat([result1, result2])

	if debug is True:

		cv2.imshow("output1", result1)
		cv2.imshow("output2", result2)
		cv2.imshow("Stacked Output", result)
		cv2.waitKey(0)

	save_file_path = os.path.join(save_folder, filename)
	cv2.imwrite(save_file_path, result)


	result_mask = cv2.hconcat([gt_mask*255, pd_mask*255])

	save_filename_mask = filename.replace(".jpg", "_mask.jpg")

	save_file_path_mask = os.path.join(save_folder, save_filename_mask)
	cv2.imwrite(save_file_path_mask, result_mask)


	
if debug is True:
	cv2.destroyAllWindows()



