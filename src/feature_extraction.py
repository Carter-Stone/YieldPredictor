# Functions to get wavelengths from .hdr, reduce the number of bands, find the mean for each spectral band

from spectral import envi
import numpy as np

def get_img_wavelengths(header: str) -> list[float]:        # Reference
    """
    Helper function for select_bands
    Extracts a list of the wavelengths from the header file

    In:
        header: linseed_a_24_07_03.hdr
    Out:
        wavelengths: list of wavelengths as floats
    """

    # Use the envi library to read the list of wavelengths
    try:
        wavelengths = envi.read_envi_header(header)['wavelength']
    except:
        print("Could not retrieve Wavelengths from header")
        return
    
    # Ensures wavelengths are floats
    # adapted code from https://www.geeksforgeeks.org/check-if-value-is-int-or-float-in-python/
    if isinstance(wavelengths, str):
        wavelengths = wavelengths.strip('{}').split(',')
        wavelengths = [float(wave.strip()) for wave in wavelengths]

    return wavelengths


def select_bands(images, band_wavelengths, band_ranges):
    """
    Dimensionality Reduction
    Reduces the number of wavelengths in the image matrix to specified bands.

    In:
        images: (dict) keys = image name, values = 3D numpy arrays
        band_wavelengths: List of wavelengths in image
    Returns:
        reduced_features: (dict) keys = image name, values = 3D numpy array
    """
    # Determine which bands to keep
    selected_bands = [
        i for i, wl in enumerate(band_wavelengths)
        if any(min_wl <= float(wl) <= max_wl for (min_wl, max_wl) in band_ranges)
    ]

    try:
        # Filter the entire dictionary
        # modified code from https://www.geeksforgeeks.org/python-filter-dictionary-values-in-heterogeneous-dictionary/
        reduced_dict = {}
        for key, val in images.items():
            reduced_dict[key]  = val[:, :, selected_bands]
    except:
        print ("Band selection failed")
        return
        

    return reduced_dict 


def get_avg_spectrum(images):
    """
    Flattens and extracts the mean reflectance accross all bands in each image.

    In:
        images : Dictionary {Name: 3D image array}
    Out:
        avg_spectrum : Dictionary {Name: array of channel intensities}
    """
    avg_dict = {}

    try:
        for image_name, cube in images.items():
            # flatten the pixels from {height, width, bands} 
            # to {pixel, bands} where each pixel is a row and each band is a collumn
            flat_image = cube.reshape(-1, cube.shape[-1]) 
            # take the mean for each band across all pixels
            # sum of collumn / number of rows
            mean_intensities = np.mean(flat_image, axis=0) # len(mean_intensities) = len(num_bands)
            avg_dict[image_name] = mean_intensities
    except:
        print("Issue reshaping features")#
        return

    return avg_dict
