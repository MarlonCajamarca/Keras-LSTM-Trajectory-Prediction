#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marlon Andres Cajamarca Vega

Path Prediction LSTM: Path Prediction RNN Inference script
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import argparse
import time

from keras.utils import HDF5Matrix
from keras import backend
from keras.models import load_model
from keras_radam import RAdam
from keras_lookahead import Lookahead

def load_datasets(in_h5_path, partition ='train'):
    """
    Load train or test dataset
    
    Parameters
    ---------
    in_h5_path -> hdf5 : Path to the train/test input dataset
    partition -> str : Parameter to choose which dataset to load. Could be either "train" or "test"
    
    Returns
    -------
    encoder_input_* -> HDF5_Group : HDF5_Group containing train/test encoder_input dataset 
    decoder_input_* -> HDF5_Group : HDF5_Group containing train/test decoder_input dataset
    decoder_target_* -> HDF5_Group : HDF5_Group containing train/test decoder_target dataset
    """
    if partition == 'train':
        X_train = HDF5Matrix(datapath=in_h5_path, dataset = "train/X_train")
        y_train = HDF5Matrix(datapath=in_h5_path, dataset = "train/y_train")
        return X_train, y_train
    elif partition == 'test':
        X_test = HDF5Matrix(datapath=in_h5_path, dataset = "test/X_test")
        y_test = HDF5Matrix(datapath=in_h5_path, dataset = "test/y_test")
        return X_test, y_test
    else:
        print("Invalid 'partition' parameter: Valid values: ['train', 'test']")

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error custom metric
    
    Parameters
    ----------
    y_true -> float : Ground-truth value for RMSE metric calculation
    y_pred -> float : Predicted value for RMSE metric calculation 
    
    Returns
    ------
    rmse_value -> float : The RMSE value between y_true and y_pred
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def load_models(lstm_model_path, model_history_path, config_file_path):
    """
    Load previuously trained model for evaluation and inference purposes
    
    Parameters
    ----------
    lstm_model_path -> str : Path to pretrained LSTM model
    model_history_path -> str : Path to pretrained LSTM model's history
    config_file_path -> str : Path to configurations file containing needed hyperparameters
    
    Returns
    ------
    lstm_model -> keras model : Keras pretrained model for the complete encoder-decoder model training
    train_history -> Json : LSTM model's history
    hyperparams -> Dict : Dictionary containing needed hyperparameters for model evaluation
    """
    lstm_model = load_model(lstm_model_path, custom_objects={'rmse': rmse, 'Lookahead': Lookahead, 'RAdam' : RAdam})
    lstm_model.summary(line_length = 180)
    # Printing the input and output model's layer names. They will be needed in the Classifier to load the LSTM models correctly.
    print("-----_model_inputs------")
    print(lstm_model.inputs)
    print("----- model_outputs------")
    print(lstm_model.outputs)
    print("---------------------")
    with open(model_history_path) as json_file:
        train_history = json.load(json_file)
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)   
    return lstm_model, train_history, hyperparams

def plot_results(results_path, history, mode = "loss"):
    """
    Plot the loss + metric training and validation results for the given model
    
    Parameters
    ----------
    results_path -> str : Path to the training results files
    mode -> str : Select the mode in order to print "loss" or "metric" results
    
    Returns : Void
    ------
    """
    if mode == "loss":
        plt.figure(figsize=(10,10))
        plt.plot(history[mode][1:])
        plt.plot(history["val_" + mode][1:])
        plt.title("Model MSE " + mode)
        plt.xlabel("# Epochs")
        plt.ylabel(mode + " score")
        plt.legend(["train", "Validation"], loc = "upper right", fontsize = 14)
        plt.savefig(results_path)
        plt.show()       
    elif mode == "metric":
        metric_name = "rmse"
        plt.figure(figsize=(10,10))
        plt.plot(history[metric_name][1:])
        plt.plot(history["val_" + metric_name][1:])
        plt.title("Model " + metric_name + " metric")
        plt.xlabel("# Epochs")
        plt.ylabel(metric_name + " score")
        plt.legend(["train", "Validation"], loc = "upper right", fontsize = 14)
        plt.savefig(results_path)
        plt.show()
    else:
        print("A valid mode must be selected! Valid modes: 'loss' or 'metric'")
    

def evaluate_models(lstm_model, X_test, y_test, hyperparams, history, out_directory):
    """
    Evaluates the previuously trained modelsFull
    
    Parameters
    ----------
    lstm_model -> keras model : A pretrained LSTM Keras model to evaluate
    X_test -> HDF5_Group : HDF5_Group containing test LSTM input trajectories
    y_test -> HDF5_Group : HDF5_Group containing test LSTM target trajectories
    history -> json : A .json file containing all information from the training process
    out_directory -> str :Path to the output directory where resulting models, graphs and history results are saved
    hyperparams -> dict : Dictionary containing all hyperparamers for model evaluation

    Returns
    ------
    test_loss -> float : Result  loss value from model evaluation procedure
    test_metric -> float : Result metric value from model evaluation procedure

    Note : Resulting loss and metric plots will be saved at out_directory
    """
    loss_plot_name = "model_loss.png"
    results_loss_path = os.path.join(out_directory, loss_plot_name)
    plot_results(results_loss_path, history, mode = "loss")
    metrics_plot_name = "model_metric.png"
    result_metric_path = os.path.join(out_directory, metrics_plot_name)
    plot_results(result_metric_path, history, mode = "metric")
    test_loss, test_metric = lstm_model.evaluate(x = X_test, y = y_test, batch_size = hyperparams["test_batch_size"], verbose = 1)
    print("MODEL EVALUATION RESULTS: \n ")
    print('Model Test loss value:', test_loss)
    print('Model Test RMSE value:', test_metric)
    return test_loss, test_metric

def predict_sequence(lstm_model, source_trajectory, n_steps_out, n_features_out):
    """
    Predicts the next n_steps_out bounding boxes using pre-trained models
    
    Parameters
    ----------
    lstm_model -> keras model : Keras LSTM model previously trained for trajectory prediction
    source_trajectory -> list : Python array containing a list of input bounding boxes forming an input trajectory to evaluate
    n_steps_out -> int : Number of bounding box predictions to perform with LSTM model
    n_features_out -> int : Number of features of the outputted bounding boxes. Bounding boxes always contains 4 features
    
    Returns
    ------
    predicted_trajectory -> list : It contains a python array with the n_steps_out predicted bounding boxes (the prediction trajectory)
    """
    src_traj_batched = np.expand_dims(source_trajectory, axis = 0)
    predicted_trajectory = lstm_model.predict(src_traj_batched)

    return np.array(predicted_trajectory)

def path_predictor(lstm_model, hyperparams, X_test, y_test, n_steps_out):
    """
    Predicts target trajectories given a list of input trajectories
    
    Parameters
    ----------
    lstm_model -> Keras LSTM model previously trained for trajectory prediction
    hyperparams -> Dict : Dictionary containing all hyperparamers for path prediction
    X_test -> HDF5_Group : HDF5_Group containing test LSTM model input trajectories
    y_test -> HDF5_Group : HDF5_Group containing test LSTM model target trajectories
    n_steps_out -> int : Number of bounding box predictions to preform with LSTM model (target sequences prediction length)
    
    Returns
    ------
    predicted_trajectories -> list : It contains a python array with the n_steps_out predicted trajectories for a set of input trajectories to evaluate
    """
    n_features_out = y_test.shape[2]  
    num_test_samples = X_test.shape[0]
    random_trajectory_samples = np.random.randint(low = 0, high = num_test_samples, size = hyperparams["num_test_predictions"])
    sorted_random_trajectory_samples = sorted(random_trajectory_samples)
    source_trajectories = X_test[sorted_random_trajectory_samples]
    target_trajectories = y_test[sorted_random_trajectory_samples]
    predicted_trajectories = []
    for idx, src_traj in enumerate(source_trajectories):
        print("Source trajectory # {}: \n {}".format(idx, src_traj))
        print("Target trajectory # {}: \n {}".format(idx, target_trajectories[idx]))
        preds = predict_sequence(lstm_model, source_trajectory = src_traj, n_steps_out = n_steps_out, n_features_out = n_features_out)
        preds = preds.astype("uint16")
        predicted_trajectories.append(preds)
        print("Predicted trajectory # {}: \n {}".format(idx, preds))

    return predicted_trajectories

def main(args):
    """
    Main function
    """
    IN_HDF5_PATH = args.dataset
    saved_models_base_path = args.output
    config_file_path = args.config
    X_test, y_test = load_datasets(IN_HDF5_PATH, partition = 'test')
    max_output_trajectory_length = X_test.shape[1]
    lstm_model_path   = os.path.join(saved_models_base_path, "lstm_model.h5")
    model_history_path = os.path.join(saved_models_base_path, "model_history.json")
    lstm_model, train_history, hyperparams = load_models(lstm_model_path, model_history_path, config_file_path)
    test_loss, test_metric = evaluate_models(lstm_model, X_test, y_test, hyperparams, train_history, saved_models_base_path)
    predicted_trajectories = path_predictor(lstm_model, hyperparams, X_test, y_test, max_output_trajectory_length)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser(description ="CLI for making Path Predictions using trained models")
    parser.add_argument("dataset", help =" Path to input test dataset in .hdf5 format")
    parser.add_argument("output", help ="Path to the output directory where resulting training graphs and prediction results are computed.")
    parser.add_argument("config", help ="Path to configuration file used for testing.")
    args = parser.parse_args()
    main(args)