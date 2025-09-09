import openslide
from glob import glob
import numpy as np
import json
import os
from tqdm import tqdm

# Directory containing WSI images
wsi_directory = "/home/txiang/pathology/data_raw/NSCLC/LUAD"

# Get all WSI image file paths
wsi_files = glob(os.path.join(wsi_directory, "*.svs"))

# Dictionary to store width and height information
wsi_info_luad = {}

# Iterate over each WSI image
for wsi_file in tqdm(wsi_files):
    # Open the WSI image
    try:
        slide = openslide.OpenSlide(wsi_file)
    except:
        print("Error: ", wsi_file)
        continue
    
    # Get the width and height
    width, height = slide.dimensions
    
    # Close the WSI image
    slide.close()
    
    # Store the width and height in the dictionary
    wsi_info_luad[os.path.basename(wsi_file).replace(".svs", "")] = {"width": width, "height": height, "type": "LUAD"}


# Path to store the JSON file
json_file = "luad.json"

# Write the dictionary to the JSON file
with open(json_file, "w") as f:
    json.dump(wsi_info_luad, f)



# Directory containing WSI images
wsi_directory = "/home/txiang/pathology/data_raw/NSCLC/LUSC"

# Get all WSI image file paths
wsi_files = glob(os.path.join(wsi_directory, "*.svs"))

# Dictionary to store width and height information
wsi_info = {}

# Iterate over each WSI image
for wsi_file in tqdm(wsi_files):
    # Open the WSI image
    try:
        slide = openslide.OpenSlide(wsi_file)
    except:
        print("Error: ", wsi_file)
        continue
    
    # Get the width and height
    width, height = slide.dimensions
    
    # Close the WSI image
    slide.close()
    
    # Store the width and height in the dictionary
    wsi_info[os.path.basename(wsi_file).replace(".svs", "")] = {"width": width, "height": height, "type": "LUSC"}


# Path to store the JSON file
json_file = "lusc.json"

# Write the dictionary to the JSON file
with open(json_file, "w") as f:
    json.dump(wsi_info, f)


# Combine the two dictionaries
wsi_info_combined = {**wsi_info_luad, **wsi_info}
json_file = "nsclc.json"
with open(json_file, "w") as f:
    json.dump(wsi_info_combined, f)

