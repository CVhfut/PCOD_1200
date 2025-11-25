# IPNet: Polarization-based Camouflaged Object Detection via dual-flow network
### Authors: Xin Wang, Jiajia Ding, Zhao Zhang, Junfeng Xu and Jun Gao.

## Downloading necessary data:
* The PCOD_1200 dataset can be found in this [download link (Google Drive)](https://drive.google.com/uc?export=download&id=1cflvU9lAHaRFppMKlD0UG4xVNTkHVh6s)

* Four images with different polarization angles can be found in this [download link (Google Drive)](https://drive.google.com/uc?export=download&id=1ykmaK9eFCJBWz7qE1TWM8-g9f0cj9WIj).
## Network Architecture
## Results and Saliency maps
## Content Description
### Training/Testing
* The training and testing experiments are conducted using PyTorch with a single NVIDIA 3090ti GPU of 24 GB Memory.
* Please run
```
python MyTest.py
```
### Code:
* unpolar_rgb.m/untitled2.m/AOP_DOP_new.m: Stokes parameter image computation in MATLAB, with input consisting of four images 

  at different polarization angles. Specific differences are detailed in the code.

* data_enhance.py: Data Augmentation, Augment the images in the dataset by horizontally and vertically flipping them.

* convert.py: Process the JSON files generated after using LabelMe for ground truth annotation for subsequent processing.

* Get_gt_from_json.py: Convert the output from the above convert.py script into a binary image (black and white).

* sal2edge:Generate object edges using Ground Truth(GT).

## Citation
```
@article{WANG2024107303,
title = {IPNet: Polarization-based Camouflaged Object Detection via dual-flow network},
journal = {Engineering Applications of Artificial Intelligence},
volume = {127},
pages = {107303},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2023.107303},
author = {Xin Wang and Jiajia Ding and Zhao Zhang and Junfeng Xu and Jun Gao},
}
```
