# DeGMix: Efficient Multi-Task Dense Prediction with Deformable and Gating Mixer

This repo is the official implementation of ["DeMTG"](https://doi.org/10.48550/arXiv.2308.05721) as well as the follow-ups. It currently includes code and models for the following tasks:



## Updates

***17/12/2023***
We release the models and code of DeMTG.


## Introduction

**DeGMix** 
We introduce an efficient multi-task dense prediction with **de**formable and **g**ating **mix**er (DeGMix), a simple and effective encoder-decoder architecture up-to-date that incorporates the convolution and attention mechanism within a unified framework for MTL.

First, the deformable mixer encoder contains two types of operators: the channel-aware mixing operator leveraged to allow communication among different channels, and the spatial-aware deformable operator with deformable convolution applied to efficiently sample more informative spatial locations.
By simply stacking the operators, we obtain the deformable mixer encoder, which effectively captures significant deformable features.
Second, the task-aware gating transformer decoder is used to perform the task-specific predictions, in which task interaction block integrated with self-attention is applied to capture task interaction features, and the task query block integrated with gating attention is leveraged to dynamically select the corresponding task-specific features.
Furthermore, the results of the experiment demonstrate that the proposed DeGMix uses fewer GFLOPs and significantly outperforms current Transformer-based and CNN-based competitive models on a variety of metrics on three dense prediction datasets (\textit{i.e.,} NYUD-v2, PASCAL-Context, and Cityscapes).
For example, using Swin-L as a backbone, our method achieves 57.55 mIoU in segmentation on NYUD-v2, outperforming the best existing method by +3.99 mIoU.
