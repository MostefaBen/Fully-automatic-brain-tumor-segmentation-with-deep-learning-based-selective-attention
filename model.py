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

from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import SGD


def build_model_SparseMultiOCM(input_shape=(64, 64, 4), kernel_size=(3, 3), load_model_weights=None, nb_classes=4):
        
        """
        Parameters
        ----------
        input_shape: the input of the model
            Shape: tuple of 3 numbers 
            Default size: (64, 64, 4)
        kernel_size: kernel of the model
            Shape: tuple of 2 numbers 
            Default size: (3, 3)
        load_model_weights: the stored parameters
             Default: None
        nb_classes: The number of classes
            Default: 4
        Returns
        -------
        model: an instance of the created keras model.
        """
    
        inputs= Input(shape=input_shape)
        
        x = Conv2D(
            filters=42,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(inputs)
        
        conv1 = Conv2D(
            filters= 42,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv1)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
       
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([conv1,x])
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        conv2 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv2)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([conv1, conv2, x])
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=nb_classes, 
            kernel_size=(1, 1),
            strides=1,
            padding='same', 
            activation='softmax')(x)
    
        model = Model(inputs=[inputs], outputs=[x])
        
        model.compile(
            optimizer=SGD(
            lr=0.001,
            decay=1e-6,
            momentum=0.9,
            nesterov=True),
            loss='categorical_crossentropy')
        
    
        if load_model_weights is not None:
            model.load_weights(load_model_weights)
            print("weights are loaded")
        else:
            print("weights are None")
        
        return model


def build_model_InputSparseMultiOCM(input_shape=(64, 64, 4), kernel_size=(3, 3), load_model_weights=None, nb_classes=4):
        
        """
        Parameters
        ----------
        input_shape: the input of the model
            Shape: tuple of 3 numbers 
            Default size: (64, 64, 4)
        kernel_size: kernel of the model
            Shape: tuple of 2 numbers 
            Default size: (3, 3)
        load_model_weights: the stored parameters
             Default: None
        nb_classes: The number of classes
            Default:4
        Returns
        -------
        model: an instance of the created keras model.
        """
    
        inputs= Input(shape=input_shape)
        
        x = Conv2D(
            filters= 42,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(inputs)
        
        conv1 = Conv2D(
            filters= 42,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv1)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
       
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([inputs,conv1,x])
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        conv2 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv2)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([inputs,conv1, conv2, x])
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=nb_classes, 
            kernel_size=(1, 1),
            strides=1,
            padding='same', 
            activation='softmax')(x)
    
        model = Model(inputs=[inputs], outputs=[x])
    
        model.compile(
            optimizer=SGD(
            lr=0.001,
            decay=1e-6,
            momentum=0.9,
            nesterov=True),
            loss='categorical_crossentropy')
        
        if load_model_weights is not None:
            model.load_weights(load_model_weights)
            print("weights are loaded")
        else:
            print("weights are None")
        
        return model
    
    
def build_model_DenseMultiOCM(input_shape=(64, 64, 4), kernel_size=(3, 3), load_model_weights=None, nb_classes=4):
        
        """
        Parameters
        ----------
        input_shape: the input of the model
            Shape: tuple of 3 numbers 
            Default size: (64, 64, 4)
        kernel_size: kernel of the model
            Shape: tuple of 2 numbers 
            Default size: (3, 3)
        load_model_weights: the stored parameters
             Default: None
        nb_classes: The number of classes
            Default: 4
        Returns
        -------
        model: an instance of the created keras model
        """
    
        inputs= Input(shape=input_shape)
        
        conv1 = Conv2D(
            filters= 32,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu')(inputs)
        
        conv2 = Conv2D(
            filters= 32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(inputs)
        
        conv2 = Conv2D(
            filters= 32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv2)
        
        conv3 = Conv2D(
            filters= 32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv2)
        
        conv3 = Conv2D(
            filters= 32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv3)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv3)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
       
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([conv1,conv2,conv3,x]) 
        
        conv1 = Conv2D(
            filters= 32,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu')(x)
        
        conv2 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        conv2 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv2)
        
        conv3 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv2)
        
        conv3 = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(conv3)
        
        x = MaxPooling2D(
            pool_size=(2,2))(conv3)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = UpSampling2D(
            size=(2,2))(x)
        
        x = concatenate([conv1, conv2, conv3, x])
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=32,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu')(x)
        
        x = Conv2D(
            filters=nb_classes, 
            kernel_size=(1, 1),
            strides=1,
            padding='same', 
            activation='softmax')(x)
    
        model = Model(inputs=[inputs], outputs=[x])
    
        model.compile(
            optimizer=SGD(
            lr=0.001,
            decay=1e-6,
            momentum=0.9,
            nesterov=True),
            loss='categorical_crossentropy')
    
        if load_model_weights is not None:
            model.load_weights(load_model_weights)
            print("weights are loaded")
        else:
            print("weights are None")
        
        return model