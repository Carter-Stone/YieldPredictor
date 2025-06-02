# predicts the yield of a new image

import pickle
import numpy as np

def predict_yield(new_image, model_path):
    """
    generate a yield prediction for the new image data

    In:

    Out:

    """

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    yield_pred = model.predict(new_image)
    return yield_pred

