#=====================================================================================================================
#Copyright 2019 Mostefa Ben naceur
#@author: Mostefa Ben naceur
#(https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention)
#@email:bennaceurmostefa@gmail.com
#@year:  2019
# All Rights Reserved
# Keras implementation of the paper: Fully automatic brain tumor segmentation with deep learning based selective 
# attention using overlapping patches and multi-class weighted cross-entropy. By Mostefa Ben naceur
#=====================================================================================================================

import numpy as np
from glob import glob
import cv2
import SimpleITK as sitk


class Prediction(object):
    
    def __init__(self, batch_size_test, load_model_weights= None):
        
         """
         Parameters
         ----------
         batch_size_test: the number of training examples 
                          utilized in one iteration
         load_model_weights: the stored parameters
             Default: None
         """
         
         self.batch_size_test  = batch_size_test
         self.model = load_model_weights
         
        
    def predict_label(self, path_image, save_path_image, post1=False, post2=False, save=False):   
        
        """
        Parameters
        ----------
        path_image: a path to the image that
                    we want to predict
        save_path_image: a path to a file that we want 
                         to save the predicted image in.
        post1: the first post-processing techniques
          Default: False
        post2: the second post-processing techniques
          Default: False
        save : is a boolean (save/ or not) the file
          Default: False
        
            
        Returns
        -------
        prediction: the predicted image using the trained model
        """
        
        scans = self.read_image(path_image)
        
        prediction = self.model.predict(scans, batch_size=self.batch_size_test, verbose=1) 
        del scans, path_image
        
        prediction = np.argmax(prediction, axis=-1)
        prediction =  prediction.astype(np.uint8)
        prediction[prediction==3]=4
        
        if post1:
            pred_image = self.connected_component(prediction, nb_slices=155, thresh=110)
        
        if post2:
            pred_image = self.border_sharpening(pred_image, nb_slices=155)
        
        if save:
            tmp_img=sitk.GetImageFromArray(pred_image)
            del pred_image
            sitk.WriteImage(tmp_img,'predictions/{}.nii.gz'.format(save_path_image) )
            del tmp_img
        
        
        return prediction
    
    def read_image(self, path_image):

        """
        Parameters
        ----------
        path_image: a path to the image that
                    we want to predict
        Returns
        -------
        imgs: a normalized 4 MRI sequences
        """

        Flair = glob(path_image + '/*_flair.nii.gz')
        T1    = glob(path_image + '/*_t1.nii.gz')
        T1c   = glob(path_image + '/*_t1ce.nii.gz')
        T2    = glob(path_image + '/*_t2.nii.gz')
         
        imgs = [Flair[0], T1[0], T1c[0], T2[0]]
        del Flair, T1, T2, T1c
        
        imgs = np.array([sitk.GetArrayFromImage(sitk.ReadImage(imgs[i])) for i in range(len(imgs))]).astype(np.float32)

        imgs = self.image_preprocessing(imgs)
        
        imgs = imgs.swapaxes(0,1)
        imgs = np.transpose(imgs,(0,2,3,1))

        return imgs
        
            
    
    
    def image_preprocessing(self, slices, size=[4, 155, 240, 240], thresh=-9):
        
         """
        Parameters
        ----------
        slices: the MRI image's slices 
        size : is the size of slices(nb_nb_channels, nb_slices, height, width)
           Default:[4, 155, 240, 240]
        thresh: a threshold to isolate the background pixels from the mean of intensities
           Default:-9
            
        Returns
        -------
        norm_slices: a normalized MRI sequences
        """
        
         norm_slices = np.zeros((size[0], size[1], size[2], size[3]))
        
         for channel in range(size[0]):
            
            for slice in range(size[1]):
                
                norm_slices[channel][slice] = self.image_normalization(slices[channel][slice], thresh)

         return norm_slices    


    def image_normalization(self, slices, thresh, noise_cap=1.0):
        
         """
        Parameters
        ----------
        slices: the MRI image's slices 
        thresh: a threshold to isolate the background pixels from the mean of intensities
        noise_cap: a percent of noise that will be removed
           Default: 1.0
            
        Returns
        -------
        norm_slices: a normalized slice
        """
        
         slices = np.clip(slices, np.percentile(slices, noise_cap), np.percentile(slices, noise_cap))
        
         img = slices[np.nonzero(slices)]
        
         img= (slices - np.mean(img))/ np.std(img)
         del slices
        
         img[img==img.min()]= thresh
        
         return img


    def connected_component(self, pred_image, nb_slices, thresh=110):
        
         """
        Parameters
        ----------
        pred_image: a predicted image by a trained model
        nb_slices:  the number of slices in a image (the depth)
        thresh: a threshold to determine of connected components  
                that we want to remove
          Default: 100
          
        Returns
        -------
        pred_image: a predicted image without some regions
        """
        
         for k in range (nb_slices):
            
            img = pred_image[k]
            img = img.astype(np.uint8)
               
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            
            sizes = stats[1:, -1]
            del stats, centroids
            nb_components = nb_components - 1
            min_size = thresh
        
            tmp_img2 = np.zeros((output.shape))
            for i in range(0, nb_components):
                    if sizes[i] >= min_size:
                        tmp_img2[output == i + 1] = 255
            
            del nb_components, min_size, sizes
            tmp_img3 = np.zeros((output.shape))   
            del output
            for i in range((img.shape[0])):
                    for j in range((img.shape[0])):
                        if(tmp_img2[i][j] > 0):
                            tmp_img3[i][j] = img[i][j]
                            
            pred_image[k] = tmp_img3  
            del tmp_img2, tmp_img3
           
            
         return pred_image       
    
    
    def border_sharpening(self, pred_image, nb_slices):
        
         """
        Parameters
        ----------
        pred_image: a predicted image by a trained model and preprocessed
                    by the first post-processing
        nb_slices:  the number of slices in a image (the depth)
        
        Returns
        -------
        pred_image: border sharpening of the predicted image using
                    a morphological opening operator
        """
        
         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
         for k in range (nb_slices):
            
            img_in = pred_image[k]
            img_in = img_in.astype(np.uint8)
            
            img_out = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, kernel)
        
            pred_image[k] = img_out  
        
         del kernel
        
         return pred_image
    