import nrrd
import numpy as np
from PIL import Image
import os

#path to file 
file_path = "C:/Users/diego/Bac Sport and Computer Science/Documents/BA-arbeit/20112024/VL_009_MZP2.nrrd"
output_folder = "output_slices"  
#create the folder
os.makedirs(output_folder, exist_ok=True)

#load the file
data, header = nrrd.read(file_path)


#save the slices in .tif
for i in range(data.shape[2]):  
    slice_data = data[:, :, i]
    #noramlize from 0 to 255
    normalized_slice = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
    normalized_slice = normalized_slice.astype(np.uint8)

    image = Image.fromarray(normalized_slice)
    output_path = os.path.join(output_folder, f"slice_{i:03d}.tif")
    image.save(output_path)

