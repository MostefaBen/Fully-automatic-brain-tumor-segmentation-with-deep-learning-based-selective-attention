#=====================================================================================================================
#Copyright 2019 Mostefa Ben Naceur
#@author: Mostefa Ben Naceur 
#(https://github.com/MostefaBen/Fully-automatic-brain-tumor-segmentation-with-deep-learning-based-selective-attention)
#@email:bennaceurmostefa@gmail.com
#@year:  2019
# All Rights Reserved
# Keras implementation of the paper: Fully automatic brain tumor segmentation with deep learning based selective 
# attention using overlapping patches and multi-class weighted cross-entropy. By Mostefa Ben naceur
#=====================================================================================================================

from keras.callbacks import  ModelCheckpoint, LearningRateScheduler,  EarlyStopping
from model import build_model_SparseMultiOCM, build_model_InputSparseMultiOCM, build_model_DenseMultiOCM
from prediction import Prediction


class Training(object):
    
    def __init__(self, model,  batch_size, epoches, load_model_weights=None, class_weighting=None):
        
         """
         Parameters
         ----------
         model: the created model
         batch_size: the number of training examples 
                    utilized in one iteration
         epoches: one epoch is when a full dataset 
                  is passed forward and backward 
                  through the neural network only once.
         load_model_weights: the stored parameters
                  Default: None
         class_weighting: a dictionary of coefficients 
                         to weight the loss contributions 
                         of different model outputs
         """
         
         self.batch_size  = batch_size
         self.epoches     = epoches
         self.class_weighting = class_weighting
        
         if load_model_weights is not None:
            self.model = load_model_weights
         else:
            self.model  = model
       
        
    def fit_model(self, X_patches, Y_labels, X_patches_valid, Y_labels_valid):
        
        """
        Parameters
        ----------
        X_patches, Y_labels: the patches and labels of a
                             training set
        X_patches_valid, Y_labels_valid: the patches and 
                                        labels of an evaluation set
        Returns
        -------
        history: a record of training/evaluation loss values and 
                 metrics values at successive epochs
        """
        
        callbacks= [                  
                     LearningRateScheduler(lambda x: 1e-3 * 0.99 ** x),
                     EarlyStopping(monitor='val_loss',  patience= 10, verbose = 1,  mode='auto'),
                     ModelCheckpoint(filepath ='brain_segmentation/.{epoch:02d}_{val_loss:.3f}.hdf5', verbose=1)
                   ]

        history = self.model.fit(X_patches, Y_labels, epochs= self.epoches, batch_size= self.batch_size, validation_data=(X_patches_valid,Y_labels_valid), verbose= 1, callbacks = callbacks,  class_weight= self.class_weighting)
        del callbacks
        
        
        return history



if __name__ == "__main__":
    
    epoches = 100
    batch_size = 8
    input_shape=(64, 64, 4)
    nb_classes= 4
    
    model= build_model_InputSparseMultiOCM(input_shape=input_shape, load_model_weights= None, nb_classes=nb_classes)
    del epoches, batch_size, input_shape, nb_classes
    
    print(model.summary())
    
    #training phase
    class_weighting= [0.28, 0.08, 0.43, 0.21]
    
    Init_train = Training(model, batch_size, epoches, class_weighting)    
    del class_weighting
    
    history = Init_train.fit_model()
    
    #prediction phase
    load_model_weights = " " #load the trained model
    batch_size_test = 8
    
    init_pred   = Prediction(batch_size_test, load_model_weights)
    del batch_size_test, load_model_weights

    path_image= " " # path to the image
    save_path_image =  " " # path to the storage file
    
    prediction  = init_pred.predict_label(path_image, save_path_image)
    del path_image, save_path_image
