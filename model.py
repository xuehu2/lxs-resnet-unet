from backbone.Resnet34 import *
from backbone.Resnet50 import *
from tensorflow.keras import layers,activations,models
import numpy as np
import tensorflow as tf


def Resnet50_Unet(batchsize=1, inputsize=(512,512,3)):
    """

    :type inputsize: object
    """
    img = layers.Input(shape=inputsize, batch_size=batchsize)

    f1,f2,f3,f4 = ResNet50(img)

    up = layers.UpSampling2D(size=(2,2))(f4) 
    up = layers.Concatenate(axis=-1)([up,f3])
    up = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)
    up = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)

    up = layers.UpSampling2D(size=(2,2))(up)
    up = layers.Concatenate(axis=-1)([up,f2])
    up = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)
    up = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)

    up = layers.UpSampling2D(size=(2,2))(up)
    up = layers.Concatenate(axis=-1)([up,f1])
    up = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)
    up = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up)

    up = layers.UpSampling2D(size=(4,4))(up)
    up = layers.Conv2D(2, 1, activation='softmax')(up)

    model = models.Model(inputs=img, outputs=up)
    return model

model = Resnet50_Unet()
model.summary()


