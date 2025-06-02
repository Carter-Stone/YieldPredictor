#load and sync HSI/CSV data
import os
import numpy as np
import pandas as pd
import spectral as sp
from glob import glob

# Loads all images from a folder to a dictionary variable
def load_hs_images(images_path):
    """
    Loads all the hyperspectral images from a directory using...

    In:
        images_path : path to the image directory.
    Out:
        image_dict : a dictionary containing 3D NumPy arrays.
    """
    image_dict = {}
    # Uses spectral library to handle ENVI format
    # Using spectral to read the .hdr file automatically decrypts the .dat file
    for folder, _, files in os.walk(images_path):
        for file in files:
            if file.endswith('.hdr'):
                new_path = os.path.join(folder, file)
                img = sp.open_image(new_path).load().astype('float32')
                key_name = os.path.splitext(os.path.basename(file))[0]                 # ??? #
                image_dict[key_name] = img
                
    return image_dict

# Creates a dataframe containing the yield data
def load_yield_data(yield_path):
    """
    Loads yield data from a csv file.

    In:
        csv_path : path to the yield file.
    Out:
        DataFrame with plot numbers and yields.
    """
    yield_data = pd.read_csv(yield_path) # dataframe
    return yield_data

if __name__ == "__main__":

    print ("loading data")

    images_path = "data/hyperspectral_images"
    yield_csv = "data/yield_data.csv"

    images = load_hs_images(images_path) # dict
    yields = load_yield_data(yield_csv)  # df 

    print(f"Images loaded: {len(images)}")
    print(f"Yield data loaded")