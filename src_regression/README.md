# Regression-based-2D-3D-Registration

## Prerequisites
Unfortunately, we cannot release the CT data, weights of the model, or the IDs due to the [Database Access, Sharing, and Use Agreement](https://nmdid.unm.edu/resources/data-use) from [NMDID](https://nmdid.unm.edu/).

Code for segmenting out the pelvis regions in the CT scan of the pelvis come from https://github.com/mkrcah/bone-segmentation.

## Preprocessing CT scans download from NMDID
For preprocessing, we have followed the steps below:
- Slice the CT scans to only include the regions where pelvis exists and make the dimension of the CT to be cubic. For example, if the scan was (512, 512, 400), we make it to (400, 400, 400) without cutting any regions in the CT scan.
- Using the sliced CT scans, segment out the bone regions using the bone segmentation from Krčah, Marcel, Gábor Székely, and Rémi Blanc. "Fully automatic and fast segmentation of the femur bone from 3D-CT images with no shape prior." 2011 IEEE international symposium on biomedical imaging: from nano to macro. IEEE, 2011.
- Remove the background of the CT scan using the segmentation
- Resize the CT scan and the segmentation to (128, 128, 128)
- Flip the CT scan and the segmentation so that it matches the orientation of the data provided as `CT128.nii` and `CTSeg128.nii` from Gao, Cong, et al. "Generalizing spatial transformers to projective geometry with applications to 2D/3D registration." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer International Publishing, 2020.

## Generating DRRs
Put your preprocessed CT scans into `../data/CT/` and segmentations to `../data/CTSeg`. For generating Digitally Reconstructed Radiographs (DRRs) using ProST:
```
python3 ProST_DRR_Generation.py
```
This will generate images for baseline and hard cases.

## Training Regression Model
For training regression model for pose estimation of the hip select one version from below:
```
python3 GuessNet_Regression_ver1.py
python3 GuessNet_Regression_ver1_PE.py
python3 GuessNet_Regression_ver1_PE_AC.py
python3 GuessNet_Regression_ver2.py
python3 GuessNet_Regression_ver2_PE.py
python3 GuessNet_Regression_ver2_PE_AC.py
```