# Lane-Marker-Generation-Using-Transformer-Based-GANs
This repository contains the code for **GLFormer**. 

## Description
Lane segmentation is a crucial component for the proper functioning of a completely
autonomous vehicle. Most of the methods that exist right now depend predominantly on
the lane markers being clearly visible. This is not always the case as there are challenging
scenarios where lane lines have been degraded due to wear and tear, and adverse weather
conditions such as snow and even dust. This projects aims to tackle this challenge by
taking a generative approach to lane segmentation based on Transformers. To this
end, this project proposes an Image-to-Image generative adversarial network comprised
of Vision Transformers to predict lane segmentation mask and improve the quality of
the generated segmentation mask by incorporating the Embedding Loss. Being able to
determine the position of the lane lines based on local and global features in the image
would be critical while driving in situation with limited lane line visibility.
