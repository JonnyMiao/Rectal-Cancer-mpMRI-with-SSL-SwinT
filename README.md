# A Deep Learning Framework with Self-Supervised Learning for Developing Multi-dimensional Prediction Model in Rectal Cancer

 This is the source code of Rectal MRI prediction model for predicting TRG, T-downstage and ypN status. The model takes as inputs the mpMRI image and five clinical features, the model is based on the [SwinTransformer](https://github.com/microsoft/Swin-Transformer).  with the cross attention mechanism.

## Installation

 This code depends on [SwinTransformer](https://github.com/microsoft/Swin-Transformer). Below are quick steps for installation. Please refer to (https://github.com/microsoft/Swin-Transformer#installation) for more detailed instruction. 

Also, please make sure you get the cuda on your computer.

pip install -r requirement.txt

## Run

 Run python main.py for the training and inference.