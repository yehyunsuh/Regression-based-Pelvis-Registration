# Regression-based-2D-3D-Registration

## Prerequisites
All the code for ProST come from 
- MICCAI: https://github.com/gaocong13/Projective-Spatial-Transformers
- TMI: https://github.com/gaocong13/Projective-Spatial-Transformers/tree/tmi

For setting up to use ProST, please follow the instructions written in the original repository. You will have to train a model for your own dataset that they have for each MICCAI and TMI version. The vanilla `pretrain.pt` model that is provided from ProST will not work on new dataset that were not used for training. You can specify the model you trained in the `--model_registration` argument for MICCAI and TMI paper.

## Registration
In order to run the registration, run one of `ProST_Intensity_base.py`, `ProST_Intensity_hard.py`, `ProST_MICCAI_base.py`, `ProST_MICCAI_hard.py`, `ProST_TMI_base.py`, `ProST_TMI_hard.py`. For each python file, you can run each of the 7 commands below to run the same experiments as the paper.
```
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input1 --model_type Input1_r18 --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1.pth
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input1_PE --model_type Input1_r18_PE --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1_PE.pth
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input1_PE_AC --model_type Input1_r18_PE_AC --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1_PE_AC.pth
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input2 --model_type Input1_r18 --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1.pth
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input2_PE --model_type Input1_r18_PE --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1_PE.pth
python3 ProST_Intensity_base.py --wandb_project ProST_Intensity_Input2_PE_AC --model_type Input1_r18_PE_AC --model_name resnet18 --weight_path ../model_weights/GuessNet_Regression_baseline_ver1_PE_AC.pth
```