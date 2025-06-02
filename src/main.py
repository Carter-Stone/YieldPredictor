
# main.py - load data, extract values, sync data, train and test model, print results

import loader
import feature_extraction
import model
import re
import numpy as np
import matplotlib.pyplot as plt

print("Hello...")

def main (
        images_path = "data/hyperspectral_images",
        yield_csv = "data/yield_data.csv"):
    
    print("Loading data...")
    try:
        images = loader.load_hs_images(images_path) # dict
        yields = loader.load_yield_data(yield_csv)  # df 
        print(f"Images loaded: {len(images)}")
        print(f"Yield data loaded")
    except:
        print("Failed to load data")


    # Convert to dictionary
    # https://www.geeksforgeeks.org/pandas-dataframe-to_dict/
    #yield_dict = yields.to_dict() # dict   
    yield_dict = dict(zip(yields["plot"].str.strip(), yields["biomass"]))

    print("Extracting features...")
    band_wavelengths = feature_extraction.get_img_wavelengths(f"{images_path}/linseed_24_07_03/linseed_a_24_07_03.hdr")
    # Bands selected based off of prior research specific to chlorophyll in Linseed
    reduced_bands = feature_extraction.select_bands(images, band_wavelengths, band_ranges = [(690, 740), (770, 800)]) # dict
    avg_spectrum = feature_extraction.get_avg_spectrum(reduced_bands) # dict

    features = []
    labels = []

    for image_name, spectrum in avg_spectrum.items():           # for each image spectrum
        match = re.match(r'(linseed_[a-e])', image_name)           # code adapted from https://developers.google.com/edu/python/regular-expressions 
        if match:                                                                                  # find plot name
            plot_id = match.group(1) # 'linseed_a-e'
            print(plot_id)
            if plot_id in yield_dict:
                features.append(spectrum) # adds the spectrum to 'features'
                labels.append(yield_dict[plot_id]) # adds the corresponding plot yield to 'labels'
                # As features and labels are appended at the same time the lengths will be the same
            else: 
                print(f"Yield not found. Image:{image_name}") 
        else:
            print(f"Plot_id not found. Image:{image_name}")
            
    X = np.array(features)
    y = np.array(labels) 

    print("Training the model...")
    metrics, X_pred, y_test = model.train_eval(X, y)  # model type = 'RF'/'SVM' 
    
    # Print Eval Metrics
    print("Model performance: ")
    print(f"R2: {metrics['r2']}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")

    # Plot predictions againt Truth
   # plt.figure(figsize=(10))
    #plt.scatter(y_test, X_pred, colour='blue', label='Predictions')
    #plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal (y = x)")
    #plt.xlabel("True yield (kg/ha)")
    #plt.ylabel("Predicted Yield (kg/ha)")
    #plt.title("Truth vs Predicted yield")
    #plt.legend()
    #plt.close()

if __name__=="__main__":
        main()
