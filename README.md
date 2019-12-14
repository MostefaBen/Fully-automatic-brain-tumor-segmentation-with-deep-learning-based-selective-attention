# Fully automatic brain tumor segmentation with deep learning based selective attention using overlapping patches and multi-class weighted cross-entropy
Keras implementation of the paper <b>Fully automatic brain tumor segmentation with deep learning-based selective attention using overlapping patches and multi-class weighted cross-entropy<b> by  Ben naceur et al.

![The model architecture](https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention/blob/master/image.png)


This repository contains Keras implementation of Deep CNNs models, where these models are designed specifically for the segmentation of brain tumors. The usage is described in the next section.

# Dependencies

- Anaconda

- Python 3.6 

- Keras

- Backend: Theano or Tensorflow

# BRATS Dataset

We have used [BRATS](https://www.med.upenn.edu/sbia/brats2018/data.html) 2018 dataset of patient’s brain with high-grade (HGG) and low-grade (LGG). Each patient’s brain image comes with 4 MRI sequences (i.e. T1, T1c, T2, FLAIR) and the ground truth
of 4 segmentation labels which are obtained manually by radiologists experts: Healthy tissue, Necrotic and
Non-Enhancing tumor, Peritumoral Edema and Enhancing core.
![4 MRI sequences](https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention/blob/master/image_1.jpg)

-Each MRI volume have 4 scanning images: T1, T1c, T2, FLAIR.

-Each MRI volume have 4 segmentation labels:

```python
Label 0: background
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
```

# Usage

1. Download the repository.

2. In your python environment, unzip the repository.

3. Please note that in this paper, we developed our proposed CNNs models based on the technique of Patch-wise approach. So, the CNNs models takes as an input patches with size 64 x 64 x 4, where 4 represent the four MRI sequences (i.e. T1, T1c, T2, FLAIR)


# Results

![Results1](https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention/blob/master/result1.png)



# Issues

If you encounter any issue or have a feedback, please don't hesitate to [raise an issue](https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention/issues/new).
