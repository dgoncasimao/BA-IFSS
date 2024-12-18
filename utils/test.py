# %%
import csv
import os
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
#from datetime import datetime
#from tqdm import tqdm
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2    
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, ConvLSTM2D, Softmax, Lambda
from tensorflow.keras import Model
from keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
#from data_utilities_v2 import *
#from IFSSNet_utilities_v2 import *

# %% Check if TensorFlow is built with GPU support
print("Built with GPU support:", tf.test.is_built_with_cuda())

# List available physical devices (CPU/GPU)
physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print(f"Number of GPUs available: {len(physical_devices)}")
    for i, gpu in enumerate(physical_devices):
        print(f"GPU {i}: {gpu}")
else:
    print("No GPU available.")