
import pandas as pd
import numpy as np
import pdb

from dill import load as dill_load
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
        
    model = load_model('model.h5', compile = False)

    scalerFile = "scaler.pkl"
    with open(scalerFile, "rb") as f:
        scaler = dill_load(f)

    df = pd.read_csv('../data/HAD.csv', sep = ',')
    #pdb.set_trace()

    # X is your features, y is your target column
    X_raw, y_raw = df.iloc[:, :-2], df.iloc[:, -2]

    X = X_raw.astype('float').astype('int')
    y_typed = y_raw.astype('float').astype('int')

    y = LabelEncoder().fit_transform(y_typed)

    X_array = np.asarray(X, dtype=float)
    X_scaled = scaler.preprocess_clinical_data(X_array)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    raw_model_probs = model.predict(X_test)

    print(raw_model_probs)