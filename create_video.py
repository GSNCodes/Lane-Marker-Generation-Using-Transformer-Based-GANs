import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


debug = False

test_root_dir = '/home/gsn/Spring-2023-Finale/Capstone/GSNDGML/DatasetCapstone/Duluth_1000/test_set'

img_folder = os.path.join(test_root_dir, "images")
gt_mask_folder = os.path.join(test_root_dir, "masks")
pd_mask_folder = os.path.join(test_root_dir, "results_l2_100_dice")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("output_test.mp4", fourcc, 4, (512*3, 512)) 

my_color_palette = [[0,0,0], [0, 255, 0]]
img_grid = []
for idx, filename in enumerate(sorted(os.listdir(img_folder))):


	print(filename)



	image_path = os.path.join(img_folder, filename)
	gt_mask_path = os.path.join(gt_mask_folder, filename.replace("jpg", "png"))
	pd_mask_path = os.path.join(pd_mask_folder, filename.replace("jpg", "png"))

	image = cv2.imread(image_path)
	image = cv2.resize(image, (512, 512))
	image_copy = cv2.putText(image.copy(), 'Input', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


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


	result1 = image * 0.8 + color_seg * 0.5
	result1 = result1.astype(np.uint8)
	result1 = cv2.putText(result1, 'Ground Truth', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

	color_seg = np.zeros((pd_mask.shape[0], pd_mask.shape[1], 3), dtype=np.uint8) # height, width, 3
	palette = np.array(my_color_palette)
	for label, color in enumerate(palette):
	    color_seg[pd_mask == label, :] = color

	result2 = image * 0.8 + color_seg * 0.5
	result2 = result2.astype(np.uint8)
	result2 = cv2.putText(result2, 'Predicted', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
	

	result = cv2.hconcat([image_copy, result1, result2])

	video.write(result)

	if debug is True:

		cv2.imshow("output1", result1)
		cv2.imshow("output2", result2)
		cv2.imshow("Stacked Output", result)
		cv2.waitKey(0)

	
video.release()	
if debug is True:
	cv2.destroyAllWindows()



