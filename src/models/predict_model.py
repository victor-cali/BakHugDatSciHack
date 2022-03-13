from keras.models import load_model
import pandas as pd
import numpy as np

def predict(path, X_to_predict):
    model = load_model(path)
    predictions = model.predict(X_to_predict, batch_size=1)
    Id = np.arange(10000,dtype = np.int32)
    Category = np.array([prediction.argmax() for prediction in predictions])
    data = np.empty((10000,2))
    data[:,0] = Id
    data[:,1] = Category
    df = pd.DataFrame(
        data = data,
        columns = ('Id', 'Category')
    )
    df = df.astype(int)
    df.to_csv('../data/processed/predictions.csv', index = False)
    return df
