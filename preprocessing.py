
import pandas as pd
import numpy as np  

import seaborn as sns
import matplotlib.pyplot as plt
import json

import matplotlib.pyplot as plt

import argparse
import os
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error





def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    # load file
    data = pd.read_json(filename)
    # Rearrange cols
    cols = pd.Series(data.wn_interpolated[0]).astype(int)
    list_arrays = list(data.spectrum_interpolated)

    data = pd.DataFrame(data=list_arrays, columns=cols, index=data.glucose).reset_index()

    data.index = data.index.set_names(['experiment_number'])
    data.reset_index(inplace=True)
    
    
    data = pd.melt(data, id_vars=['glucose','experiment_number'], value_vars=data.iloc[:,2:].columns)
    data.experiment_number = data.experiment_number.apply(lambda x: x+1) 

    return data


def preprocess(data: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    features = ['variable','value']
    dicts = data[features].to_dict(orient='records')
    dicts
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def run(data_path: str, data_dest: str):
    # load json file
    data_path = os.path.join("./dataset.json")
    data = read_dataframe(data_path)

    data_train = data[data.experiment_number <= data.experiment_number.max()*0.70]
    data_val = data[(data.experiment_number > data.experiment_number.max()*0.70) & (data.experiment_number <= data.experiment_number.max()*0.85)]
    data_test = data[data.experiment_number > data.experiment_number.max()*0.85]   
    
    # extract the target
    target = 'glucose'
    y_train = data_train[target].values
    y_val = data_val[target].values
    y_test = data_test[target].values

        # fit the dictvectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(data_train, dv, fit_dv=True)
    X_val, _ = preprocess(data_val, dv, fit_dv=False)
    X_test, _ = preprocess(data_test, dv, fit_dv=False)

    # create dest_path folder unless it already exists
 
 
    # Leaf directory
    directory = "dest_path"
    
    # Parent Directories
    parent_dir = "./models"
    
    # Path
    path = os.path.join(parent_dir, directory)
    
    # Create the directory
    os.makedirs(path)

    # save dictvectorizer and datasets
    dump_pickle(dv, os.path.join(path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(path, "test.pkl"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="the location where the raw data is was saved"
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()


    run(args.raw_data_path, args.dest_path)