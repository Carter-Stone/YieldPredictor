# Train test split, Model selection, Model training, Testing and Evaluation

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import pickle


def train_eval(features, yields, model_type = 'RF', kf = True, model_path = "models/saved_model.pkl"):
    """
    Train and evaluate a machine learning model
    on the image data.

    In:
        features: The np.array containing the reduced image spectrums
        yields: np.array containing the rescaled yield list
        model_type: SVR or RF
        model_path: Storage location for pickled model
    Out:
        Metrics: RMSE, MAE snd R2 scoes after testing
        X_pred: Yield predictions for graph
        y_test: Yield truths for graph

    """
    
    # Training
    if (model_type == 'RF'):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    elif (model_type == 'SVR'):
        model = SVR(kernel='rbf', C=10)
    else: 
        print("Model ID not accepted")

    # K-Fold cross validation 
    if kf:
        K_fold = KFold(n_splits=5, shuffle=True, random_state=42)
        X_pred = cross_val_predict(model, features, yields, cv=K_fold )

        # Evaluation
        rmse = root_mean_squared_error(yields, X_pred)
        mae = mean_absolute_error(yields, X_pred)
        r2 = r2_score(yields, X_pred)

        model.fit(features, yields) # Training
        with open(model_path, 'wb') as file:
            pickle.dump((model), file)

        return {'rmse' : rmse, 'mae' : mae, 'r2' : r2}, X_pred, yields

    else:
        # Basic train test split.
        X_train, X_test, y_train, y_test = train_test_split(features, yields, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        X_pred = model.predict(X_test)  # Yield prediction

        # Evaluation
        rmse = root_mean_squared_error(y_test, X_pred)
        mae = mean_absolute_error(y_test, X_pred)
        r2 = r2_score(y_test, X_pred)

    # Saving model
    # code adapted from https://wiki.python.org/moin/UsingPickle
    with open( model_path, 'wb') as file:   #opens the file in write bit mode
        pickle.dump((model), file)  #serializes and saves the model and scaler
    
    return {'rmse' : rmse, 'mae' : mae, 'r2' : r2}, X_pred, y_test
