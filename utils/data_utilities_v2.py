#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division, absolute_import, unicode_literals


# In[ ]:


__idea__ = 'IFSS-Net'
__author__ = 'Dawood AL CHANTI'
__affiliation__ = 'LS2N-ECN'


# In[ ]:


from PIL import Image
import numpy as np
import glob
import cv2
import os
from os.path import join
import random
from medpy.io import load


# In[ ]:


def crop_pad(img):    
    H,W = img.shape
    if H>=512 and W>=512:
        PadEdgesSize1_H = int(abs(512-H)/2.)
        PadEdgesSize2_H = int(abs(512-H) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-W)/2.)
        PadEdgesSize2_W = int(abs(512-W) - PadEdgesSize1_W)
        new=img[PadEdgesSize1_H:H-PadEdgesSize2_H,PadEdgesSize1_W:W-PadEdgesSize2_W]
        
    elif H<512 and W>=512:
        new = np.vstack((img[:,:512],np.zeros_like(img[:,:512])))
        HH,WW = new.shape
        PadEdgesSize1_H = int(abs(512-HH)/2.)
        PadEdgesSize2_H = int(abs(512-HH) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-WW)/2.)
        PadEdgesSize2_W = int(abs(512-WW) - PadEdgesSize1_W)
        new=new[:512,PadEdgesSize1_W:WW-PadEdgesSize2_W]
    
    elif H>=512 and W<512:    
        new = np.hstack((img[:512,:],np.zeros_like(img[:512,:])))
        HH,WW = new.shape
        PadEdgesSize1_H = int(abs(512-HH)/2.)
        PadEdgesSize2_H = int(abs(512-HH) - PadEdgesSize1_H)
        PadEdgesSize1_W = int(abs(512-WW)/2.)
        PadEdgesSize2_W = int(abs(512-WW) - PadEdgesSize1_W)
        new=new[PadEdgesSize1_H:HH-PadEdgesSize2_H,:512]
    
    elif H<512 and W<512:   
        new = np.hstack((img[:,:],np.zeros_like(img[:,:])))
        new = np.vstack((new[:,:512],np.zeros_like(new[:,:512])))
        new = new[:512,:512]
    return new


# In[ ]:

def ReturnIndicesOfFullMask(sequence):
    '''
    The input sequence is of shape (1583, 512, 512)
    The output is a list of 2 indices, the begining and the end of thesequence with full masks
    Return the begining and the end of the sequence
    '''
    result = list(map(lambda img: img.sum() ,sequence ))
    resultIndex = list(map(lambda element: element>2950 ,result))
    Indices = [i for i, x in enumerate(resultIndex) if x]
    return Indices[0],Indices[-1]


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);
    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val
        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
    mean_iou = np.mean(I / U)
    return mean_iou



def process_mask(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    # do some pre_processing : resize into 640 x 512 x 1574
    #Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,640),
         #                                      interpolation = cv2.INTER_AREA),
                   #mask_data.transpose(2,0,1)))
            
#     Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,512),
#                                               interpolation = cv2.INTER_AREA),
#                    mask_data.transpose(2,0,1)))#,(512,640),

    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1)))#,(512,640),
    
    
    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
        #           mask_data.transpose(2,0,1)))
    
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))
    mask_gl= np.array(list(map(lambda mask_time_step:np.clip(np.where(mask_time_step == 200, mask_time_step, 0), 
                                                             0, 1,np.where(mask_time_step == 200, mask_time_step, 
                                                                           0)),Mask)))
    mask_gm= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 150, 
                                                                       mask_time_step, 0), 0, 1,
                                                              np.where(mask_time_step == 150, mask_time_step, 0)),
                               Mask)))
    
    mask_sol = np.expand_dims(mask_sol,-1)
    mask_gl = np.expand_dims(mask_gl,-1)
    mask_gm = np.expand_dims(mask_gm,-1)
    
    
    # return the whole muscles on channel axis of order SOL GL and GM
    return np.concatenate([mask_sol,mask_gl,mask_gm],-1)


# In[ ]:


#Only for SOL (Foregorund and Background)
def process_mask_SOL(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1)))#,(512,640),
    
   #resize(mask_time_step,(512,640),

    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
                   #mask_data.transpose(2,0,1)))
        
        
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))

    mask_sol = np.expand_dims(mask_sol,-1)
 
    # Get the back ground of the annotated mask using the foreground annotation
    mask_sol=np.concatenate([mask_sol,1-mask_sol],-1)

    return mask_sol


# In[ ]:


def process_data(data_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    image_data,_ = load(data_file_path)
    # Adjust the formate to (640, 528, 1574)
    image_data = image_data.transpose([1,0,2])

    image_data= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   image_data.transpose(2,0,1))) #,(512,640),
    
    #image_data= list(map(lambda image_data_step: image_data_step[:640,:512],image_data.transpose(2,0,1)))    
    
    return np.array(image_data)


# In[ ]:


def Pull_data_from_path(path):
    data = process_data(path)
    # return normalized data
    # values from whole data
    mean_val = 19.027262640214904
    std_val = 34.175155632916
    
    data = (data-mean_val) / std_val
    # reshape to t,h,w,1
    return  np.expand_dims(data,-1)


# In[ ]:


def Pull_data_from_path_Complete(path):
    data = process_data(path)
    # return normalized data
    # values from whole data
    mean_val = 19.027262640214904
    std_val = 34.175155632916
    
    data = (data-mean_val) / std_val
    # reshape to t,h,w,1
    return  np.expand_dims(data,-1)


# In[ ]:


#Only for SOL (Foregorund and Background)
def process_mask_SOL_Complete(mask_file_path):
    '''
    Return 3 masks in order SOL, GL and GM.
    Input: main path for the dataset: either Train/Val/or Testm i.e. '/tf/volumes/train/CAT_TH/masksX1.mha'
    '''
    
    #Read the data of formate (528, 640, 1574)
    mask_data,_ = load(mask_file_path)
    # Adjust the formate to (640, 528, 1574)
    mask_data = mask_data.transpose([1,0,2])

    # do some pre_processing : resize into 640 x 512 x 1574
    #Mask= list(map(lambda mask_time_step: cv2.resize(mask_time_step,(512,640),
         #                                      interpolation = cv2.INTER_AREA),
                   #mask_data.transpose(2,0,1)))
            
    Mask= list(map(lambda mask_time_step:  crop_pad(mask_time_step),
                   mask_data.transpose(2,0,1))) #(512,640)

    #Mask= list(map(lambda mask_time_step: mask_time_step[:640,:512],
               #    mask_data.transpose(2,0,1)))
    
    # get the output for each muscle of formate timesteps x H x W 
    # data clip to replace values of 100 150 and 200 to 1
    mask_sol= np.array(list(map(lambda mask_time_step: np.clip(np.where(mask_time_step == 100, mask_time_step, 0), 
                                                               0, 1,
                                                               np.where(mask_time_step == 100, mask_time_step, 0))
                                ,Mask)))

    mask_sol = np.expand_dims(mask_sol,-1)
 
    # Get the back ground of the annotated mask using the foreground annotation
    mask_sol=np.concatenate([mask_sol,1-mask_sol],-1)

    return mask_sol


# In[ ]:


def Pull_mask_from_path(path):
    return process_mask(path)


# In[ ]:


def Patient_name(diretoryPathforOnePatient):
    return diretoryPathforOnePatient.split('/')[4]


# File Format Requirements:
# 1. Input Data Structure:
#    - The data should be organized in a directory structure as follows:
#      raw_data/
#        |- <patient_number>/
#        |   |- <mzp_folder>/
#        |       |- VL_<mzp_folder>_<patient_number>.nrrd
#        |       |- Segmentation_VL_<mzp_folder>_<patient_number>.seg.nrrd
#    - Example:
#      raw_data/
#        |- 001/
#        |   |- MZP1/
#        |       |- VL_MZP1_001.nrrd
#        |       |- Segmentation_VL_MZP1_001.seg.nrrd
#        |   |- MZP2/
#        |       |- VL_MZP2_001.nrrd
#        |       |- Segmentation_VL_MZP2_001.seg.nrrd
#
# 2. File Naming Conventions:
#    - Volume file: `VL_<mzp_folder>_<patient_number>.nrrd`
#    - Mask file: `Segmentation_VL_<mzp_folder>_<patient_number>.seg.nrrd`
#
# 3. Output Data Structure:
#    - The results will be saved in a mirrored directory structure under `analyzed_data`:
#      analyzed_data/
#        |- <patient_number>/
#        |   |- <mzp_folder>/
#        |       |- volume/
#        |       |   |- volume_000.tif
#        |       |   |- volume_001.tif
#        |       |- mask/
#        |           |- mask_000.tif
#        |           |- mask_001.tif
#
# Usage:
# - To process all patients in the `raw_data` directory, use the batch processing function.
# - To process a single patient, provide the exact file paths for the volume and mask files.
 
import os
import nrrd
import numpy as np
from PIL import Image
import shutil

def crop_image_get_bounds(image, threshold):
    """
    Calculate cropping bounds dynamically based on the threshold.
    Args:
    - image: 2D numpy array to crop
    - threshold: minimum pixel intensity to consider as non-black
    Returns:
    - cropping bounds (rmin, rmax, cmin, cmax) or None if the image is completely black
    """

    rows = np.any(image > threshold, axis=1)
    cols = np.any(image > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):  # Check if the image is black
        return None

    # Find significant borders
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def apply_crop(image, bounds):
    """
    Apply cropping bounds to the image.
    Args:
    - image: 2D numpy array to crop
    - bounds: cropping bounds (rmin, rmax, cmin, cmax)
    Returns:
    - Cropped image or None if bounds are invalid
    """
    if bounds is None:
        return None
    rmin, rmax, cmin, cmax = bounds
    return image[rmin:rmax+1, cmin:cmax+1]

def resize_with_padding(image, target_size=(512, 512)):
    """
    Resize the image with padding to maintain aspect ratio.
    
    Args:
    - image: 2D numpy array (grayscale image)
    - target_size: Desired output size (height, width)

    Returns:
    - Padded image as a 2D numpy array
    """
    old_size = image.shape  # Original size (height, width)
    desired_size = target_size

    # Compute padding
    delta_w = max(desired_size[1] - old_size[1], 0)
    delta_h = max(desired_size[0] - old_size[0], 0)
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Apply padding with black (0)
    padded_image = np.pad(image, ((top, bottom), (left, right)), mode='constant', constant_values=0)

    return padded_image


def export_to_tif_with_threshold(volume_data, mask_data, output_dir_volume, output_dir_mask, threshold=10):
    """
    Exports a 3D array to .tif files, one slice per file, after cropping and normalization.
    Args:
    - volume_data: 3D numpy array of the volume
    - mask_data: 3D numpy array of the mask
    - output_dir_volume: folder to save the volume slices
    - output_dir_mask: folder to save the mask slices
    - threshold: pixel intensity threshold for black image filtering
    """
    exported_count = 0  # Count valid slices

    for i in range(volume_data.shape[2]):  # Loop through each slice
        slice_volume = volume_data[:, :, i]
        slice_mask = mask_data[:, :, i]

        # Apply the mask
        masked_volume = slice_volume * slice_mask

        # Check if the masked volume is black
        if not np.any(masked_volume > threshold):
            continue  
        
        # Calculate cropping bounds from the volume slice
        bounds = crop_image_get_bounds(slice_volume, threshold)

        # Crop both volume and mask using the same bounds
        cropped_volume = apply_crop(slice_volume, bounds)
        cropped_mask = apply_crop(slice_mask, bounds)
        
        # Skip if the cropped slice is black or invalid
        if cropped_volume is None or cropped_mask is None:
            continue
        
        # Normalize the slices to range [0, 255]
        volume_normalized = ((cropped_volume - np.min(cropped_volume)) /
                            (np.ptp(cropped_volume) + 1e-5) * 255).astype(np.uint8)
        mask_normalized = ((cropped_mask - np.min(cropped_mask)) /
                          (np.ptp(cropped_mask) + 1e-5) * 255).astype(np.uint8)
        
        # Normalize the slices to range [0, 255]
        volume_normalized = ((slice_volume - np.min(slice_volume)) /
                            (np.ptp(slice_volume) + 1e-5) * 255).astype(np.uint8)
        mask_normalized = ((slice_mask - np.min(slice_mask)) /
                          (np.ptp(slice_mask) + 1e-5) * 255).astype(np.uint8)
        
        # Resize with padding to 512x512
        volume_resized = resize_with_padding(volume_normalized, target_size=(512, 512))
        mask_resized = resize_with_padding(mask_normalized, target_size=(512, 512))

        # Save slices as .tif files
        volume_path = os.path.join(output_dir_volume, f"frame_{exported_count:05d}.tif")
        mask_path = os.path.join(output_dir_mask, f"frame_{exported_count:05d}.tif")
        Image.fromarray(volume_resized).save(volume_path)
        Image.fromarray(mask_resized).save(mask_path)

        exported_count += 1
import traceback 
def process_raw_data(raw_data_path, analyzed_data_path, threshold=10):
    """
    Process all patient subdirectories to crop and export volume and mask data.
    Args:
    - raw_data_path: base path of the raw data directory *The segmentation and the volume need to be saved in the following way: VL_MZP#_patientnumber.nrrd, Segmentation_VL_MZP#_patientnumber.seg.nrrd*
    - analyzed_data_path: base path for the output directory
    - threshold: pixel intensity threshold for black image filtering
    """
    for patient_folder in sorted(os.listdir(raw_data_path)):
        patient_path = os.path.join(raw_data_path, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for mzp_folder in sorted(os.listdir(patient_path)):
            mzp_path = os.path.join(patient_path, mzp_folder)
            if not os.path.isdir(mzp_path):
                continue

            # Construct file names dynamically using patient number and MZP folder
            # Extract numeric part and format as 00#
            patient_number = patient_folder
            volume_file = os.path.join(mzp_path, f"VL_{mzp_folder}_{patient_number}.nrrd")
            mask_file = os.path.join(mzp_path, f"Segmentation_VL_{mzp_folder}_{patient_number}.seg.nrrd")

            if os.path.exists(volume_file) and os.path.exists(mask_file):
                # Read the volume and mask data
                volume_data, _ = nrrd.read(volume_file)
                mask_data, _ = nrrd.read(mask_file)

                # Define output folder for the current MZP
                output_folder = analyzed_data_path

                # Convert `00#` back to `Patient#`
                patient_folder_name = f"Patient{int(patient_number)}"
                # Export to .tif
                output_volume_dir = os.path.join(output_folder, "volume", patient_folder_name, mzp_folder)
                if not os.path.exists(output_volume_dir):
                    os.makedirs(output_volume_dir)
                output_mask_dir = os.path.join(output_folder, "mask", patient_folder_name, mzp_folder)
                if not os.path.exists(output_mask_dir):
                    os.makedirs(output_mask_dir)
                
                
                try:
                    export_to_tif_with_threshold(volume_data, mask_data, output_volume_dir, output_mask_dir, threshold=threshold)
                except Exception as e:
                    print(patient_number, mzp_folder)
                    traceback.print_exc()

import os
import nrrd
import numpy as np
from PIL import Image
import shutil

#implement a way to outout the compact
def process_single_patient(patient_number, mzp_folder, volume_file_path, mask_file_path, analyzed_data_path, threshold=10):
    """
    Process a single patient's MZP folder to crop and export volume and mask data.
    Args:
    - patient_number: the patient ID (e.g., '001')
    - mzp_folder: the MZP folder name (e.g., 'MZP1')
    - volume_file_path: path to the volume .nrrd file
    - mask_file_path: path to the mask .nrrd file
    - analyzed_data_path: base path for the output directory
    - threshold: pixel intensity threshold for black image filtering
    """
    if not os.path.exists(volume_file_path) or not os.path.exists(mask_file_path):
        raise FileNotFoundError("Volume or mask file does not exist.")

    # Read the volume and mask data
    volume_data, _ = nrrd.read(volume_file_path)
    mask_data, _ = nrrd.read(mask_file_path)

    # Define output folder for the current MZP
    output_folder = analyzed_data_path
    
    # Convert `00#` back to `Patient#`
    patient_folder_name = f"Patient{int(patient_number)}"
    # Export to .tif
    output_volume_dir = os.path.join(output_folder, "volume", patient_folder_name, mzp_folder)
    output_mask_dir = os.path.join(output_folder, "mask", patient_folder_name, mzp_folder)
    if not os.path.exists(output_volume_dir):
        os.makedirs(output_volume_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    
    export_to_tif_with_threshold(volume_data, mask_data, output_volume_dir, output_mask_dir, threshold=threshold)

import nrrd
import h5py
import os
import argparse

def convert_nrrd_to_h5(nrrd_file):
    # Read NRRD file
    data, header = nrrd.read(nrrd_file)
    
    # Generate output filename
    h5_file = os.path.splitext(nrrd_file)[0] + ".h5"

    # Write to HDF5 file
    with h5py.File(h5_file, "w") as h5f:
        h5f.create_dataset("data", data=data)
        
        # Store metadata as attributes
        metadata = h5f.create_group("metadata")
        for key, value in header.items():
            metadata.attrs[key] = str(value)  # Convert values to string to avoid type issues

#From Gerardo
#Script that exports tif into npz for TransUnet
import os
import cv2
import numpy as np
import json
from scipy.ndimage import uniform_filter, gaussian_filter

def export_tif_to_npz(input_tif_volume_folder, input_tif_mask_folder, original_height, original_width):
    
    output_npz_folder = '../data/npz_files/augmented_train'  # Carpeta para guardar los archivos .npz
    annotations_file = './videos_first_batch_json.json'
    
    os.makedirs(output_npz_folder, exist_ok=True)  # Crear la carpeta de salida si no existe
    
    target_height, target_width = 512, 512
    

    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    pad_x = (target_width - new_width) // 2  # Padding en los bordes laterales
    pad_y = 0  # Padding en los bordes superior e inferior

    # Función sigmoide para transformar en mapas de probabilidad
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Función para normalizar al rango [0, 1]
    def normalize_minmax(x):
        return x.astype(np.float32) / 255.0

    # Función para normalizar usando z-score
    def normalize_zscore(x):
        mean = np.mean(x)
        std = np.std(x)
        # Evitar división por cero
        return (x - mean) / (std + 1e-8)

# Función para normalización logarítmica
    def normalize_log(x):
        return np.log1p(x.astype(np.float32))  # log(1 + x) para evitar log(0)

# Función para normalización por recorte de percentiles
    def normalize_clipping(x, lower_percentile=1, upper_percentile=99):
        lower = np.percentile(x, lower_percentile)
        upper = np.percentile(x, upper_percentile)
        x_clipped = np.clip(x, lower, upper)
        return (x_clipped - lower) / (upper - lower + 1e-8)

# Función para Local Contrast Normalization (LCN)
    def normalize_local_contrast(x, filter_size=15):
        local_mean = uniform_filter(x, size=filter_size)
        local_std = np.sqrt(uniform_filter(x**2, size=filter_size) - local_mean**2)
        return (x - local_mean) / (local_std + 1e-8).astype(np.float32)



    # Contador global para nombres de archivos
    frame_counter = 1

    # Parámetro para elegir el tipo de normalización ('minmax', 'zscore', 'log', 'clipping', 'local_contrast', 'none')
    normalization_type = "zscore"  # Cambiar a otro método según sea necesario

    # Iterar por cada archivo en la carpeta de imágenes
    for image_name in os.listdir(input_tif_mask_folder):
        image_path = os.path.join(input_tif_volume_folder, image_name)
        mask_path = os.path.join(input_tif_mask_folder, image_name)

    # Verificar que ambos archivos existan
        if not os.path.isfile(image_path):
            print(f"Falta la imagen: {image_path}")
            continue
        if not os.path.isfile(mask_path):
            print(f"Falta la máscara: {mask_path}")
            continue

    # Leer la imagen y la máscara
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises

        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            continue
        if mask is None:
            print(f"No se pudo leer la máscara: {mask_path}")
            continue

    # Normalizar las imágenes según el método elegido
        if normalization_type == "minmax":
            image = normalize_minmax(image)
            mask = normalize_minmax(mask)
        elif normalization_type == "zscore":
            image = normalize_zscore(image)
            mask = normalize_zscore(mask)
        elif normalization_type == "log":
            image = normalize_log(image)
            mask = normalize_log(mask)
        elif normalization_type == "clipping":
            image = normalize_clipping(image)
            mask = normalize_clipping(mask)
        elif normalization_type == "local_contrast":
            image = normalize_local_contrast(image)
            mask = normalize_local_contrast(mask)
        elif normalization_type == "none":
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)
        else:
            print(f"Tipo de normalización desconocido: {normalization_type}")
            break

        #frame_key = frame_key = image_name.replace(".tif", "")
        #scaled_coords = coords_dict.get(frame_key, np.array([]))
    # Transformar en mapas de probabilidad (opcional)
    #image_prob = sigmoid(image)  # Aplicar sigmoide a la imagen
    #mask_prob = sigmoid(mask)    # Aplicar sigmoide a la máscara

    # Generar un nombre único para cada archivo .npz en formato frame_0001, frame_0002, etc.
        output_npz_name = f"frame_{frame_counter:04d}.npz"
        output_npz_path = os.path.join(output_npz_folder, output_npz_name)
    # Guardar el archivo .npz con claves "image" y "label"

    #print("Ejemplo de frame_key en coords_dict:", list(coords_dict.keys())[:5])  # Claves en coords_dict
    #print("Ejemplo de nombres de imágenes:", os.listdir(images_folder)[:5])  # Nombres de archivos en images_folder


        try:
            np.savez_compressed(output_npz_path, image=image, label=mask)  # Agregar _prob
            #print(f"Archivo guardado: {output_npz_name}")
        except Exception as e:
            print(f"Error al guardar el archivo {output_npz_name}: {e}")
            continue

        # Incrementar el contador
        frame_counter += 1

    #print("Proceso completado.")

    #data = np.load("./npz_files/train/frame_0001.npz")
    #print("Imagen shape:", data["image"].shape)
    #print("Máscara shape:", data["label"].shape)
    #print("Coordenadas escaladas:", data["insertion_coords"])





