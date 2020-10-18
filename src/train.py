# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:38:02 2020

@author: Sai Anil Kumar M
"""

import os

import config

import argparse

import model_dispatcher

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    
    # training data is where our fold value is not equal 
    # to the kfold column in dataframe
    # we should also reset the indexes
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # validation is where the kfold is equal to the fold number
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # make X and y and then numpy arrays
    x_train = df_train.drop("target",axis=1).values
    y_train = df_train.target.values
    
    # similarly for validation
    x_valid = df_valid.drop("target", axis=1).values
    y_valid = df_valid.target.values
    
    # initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]
    
    # fir the model on training data
    clf.fit(x_train, y_train)
    
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    
    # save the model
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT, f "dt_{fold}.bin"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    
    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument(
    "--fold",
    type=int
    )
    
    parser.add_argument(
    "--model",
    type=str
    )
    
    # read the arguments from the command line
    args = parser.parse_args()
    
    # run the fold specified by command line arguments
    run(fold=args.fold,
        model=args.model)
