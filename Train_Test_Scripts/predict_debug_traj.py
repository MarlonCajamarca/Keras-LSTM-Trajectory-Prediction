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

def load_datasets(in_h5_path, partition ='test'):
    """
    Load train or test dataset
    
    Parameters
    ---------
    in_h5_path -> hdf5 : Path to the train/test insput dataset
    partition -> str : Parameter to choose which dataset to load. Could be either "train" or "test"
    
    Returns
    -------
    encoder_input_* -> HDF5_Group : HDF5_Group containing train/test encoder_input dataset 
    decoder_input_* -> HDF5_Group : HDF5_Group containing train/test decoder_input dataset
    decoder_target_* -> HDF5_Group : HDF5_Group containing train/test decoder_target dataset
    """
    if partition == 'train':
        encoder_input_train = HDF5Matrix(datapath=in_h5_path, dataset ="train/encoder_in")
        decoder_input_train = HDF5Matrix(datapath=in_h5_path, dataset = "train/decoder_in")
        decoder_target_train = HDF5Matrix(datapath=in_h5_path, dataset="train/decoder_target")
        return encoder_input_train, decoder_input_train, decoder_target_train
    elif partition == 'test':
        encoder_input_test = HDF5Matrix(datapath=in_h5_path, dataset="test/encoder_in")
        decoder_input_test = HDF5Matrix(datapath=in_h5_path, dataset="test/decoder_in")
        decoder_target_test = HDF5Matrix(datapath=in_h5_path, dataset="test/decoder_target")
        return encoder_input_test, decoder_input_test, decoder_target_test
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

def load_models(train_model_path, encoder_model_path, decoder_model_path, model_history_path, config_file_path):
    """
    Load previuously trained model for evaluation and inference purposes
    
    Parameters
    ----------
    train_model_path -> str : Path to complete pretrained model
    encoder_model_path -> str : Path to pretrained inference encoder
    decoder_model_path -> str : Path to pretrained inference decoder
    
    Returns
    ------
    full_model -> keras model : Keras pretrained model for the complete encoder-decoder model training
    encoder_model -> keras model : Keras pretrained model for the inference encoder
    decoder_model -> keras_model : Keras pretrained model for the inference decoder
    """
    full_model = load_model(train_model_path, custom_objects={'rmse': rmse})
    encoder_model = load_model(encoder_model_path)
    decoder_model = load_model(decoder_model_path)
    full_model.summary(line_length = 180)
    encoder_model.summary(line_length = 180)
    decoder_model.summary(line_length = 180)
    print("----- Full_model_inputs------")
    print(full_model.inputs)
    print("----- Full_model_outputs------")
    print(full_model.outputs)
    print("----- encoder_model_inputs------")
    print(encoder_model.inputs)
    print("----- encoder_model_outputs------")
    print(encoder_model.outputs)
    print("----- decoder_model_inputs------")
    print(decoder_model.inputs)
    print("----- decoder_model_outputs------")
    print(decoder_model.outputs)
    print("---------------------")
    with open(model_history_path) as json_file:
        train_history = json.load(json_file)
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)   
    return full_model, encoder_model, decoder_model, train_history, hyperparams

def plot_results(results_path, history, mode = "loss"):
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
    

def evaluate_models(full_model, encoder_input_test, decoder_input_test, decoder_target_test, hyperparams, history, out_directory):
    """
    Evaluates the previuously trained modelsFull
    
    Parameters
    ----------
    full_model -> keras model : A previously trained keras model
    encoder_input_test -> HDF5_Group : HDF5_Group containing test encoder_input dataset
    decoder_input_test -> HDF5_Group : HDF5_Group containing test decoder_input dataset
    decoder_target_test -> HDF5_Group : HDF5_Group containing test decoder_targets dataset
    history -> json : A .json file containing all information form the previously training process
    out_directory -> str :Path to the output directory where resulting models, graphs and history results are saved
    hyperparams -> dict : Dictionary containing all hyperparamers for training/testing/predicting
        test_batch_size -> int : Number of the batch_size for the model evaluation.
    Returns
    ------
    test_loss_value -> float : Result "mse" loss value from model evaluation procedure
    test_metric_value -> float : Result "RMSE" metric value from model evaluation procedure
    Note : Resulting loss and metric plots will be saved at out_directory
    """
    loss_plot_name = "model_loss.png"
    results_loss_path = os.path.join(out_directory, loss_plot_name)
    plot_results(results_loss_path, history, mode = "loss")
    metrics_plot_name = "model_metric.png"
    result_metric_path = os.path.join(out_directory, metrics_plot_name)
    plot_results(result_metric_path, history, mode = "metric")
    test_loss, test_metric = full_model.evaluate(x = [encoder_input_test, decoder_input_test], y = decoder_target_test, batch_size = hyperparams["test_batch_size"], verbose = 1)
    print("MODEL EVALUATION RESULTS: \n ")
    print('Model Test loss value:', test_loss)
    print('Model Test RMSE value:', test_metric)
    return test_loss, test_metric

def predict_sequence(encoder_model, decoder_model, source_trajectory, n_steps_out, n_features_out):
    """
    Predicts the next n_steps_out bounding boxes using pre-trained models
    
    Parameters
    ----------
    encoder_model -> keras model : Keras model for the inference encoder
    decoder_model -> keras_model : Keras model for the inference decoder
    source_trajectory -> list : Python array containing a list of input bounding boxes
    n_steps_out -> int : Number of bounding box predictions to preform with decoder_model
    n_features_out -> int : Number of features of the outputted bounding boxes. Bounding boxes always contains 4 features
    
    Returns
    ------
    predicted_trajectory -> list : It contains a python array with the n_steps_out predicted bounding boxes
    """
    sos_token = 9999
    state = encoder_model.predict(source_trajectory)
    target_trajectory = np.array([sos_token, sos_token, sos_token, sos_token]).astype("uint16").reshape(1,1, n_features_out)
    predicted_trajectory = list()
    for _ in range(n_steps_out):
        prediction, h, c = decoder_model.predict([target_trajectory] + state)
        predicted_trajectory.append(prediction[0, 0, :])
        state = [h, c]
        target_trajectory = prediction
        time.sleep(0.5)
    return np.array(predicted_trajectory)

def path_predictor(encoder_model, decoder_model, hyperparams, encoder_input_test, decoder_target_test, n_steps_out):
    """
    Predicts the next n_steps_out bounding boxes using pre-trained models
    
    Parameters
    ----------
    encoder_model -> keras model : Keras model for the inference encoder
    decoder_model -> keras_model : Keras model for the inference decoder
    source_trajectory -> list : Python array containing a list of input bounding boxes
    encoder_input_test -> HDF5_Group : HDF5_Group containing test encoder_input dataset
    decoder_target_test -> HDF5_Group : HDF5_Group containing test decoder_targets dataset
    n_steps_out -> int : Number of bounding box predictions to preform with decoder_model
    
    Returns
    ------
    predicted_trajectories -> list : It contains a python array with the n_steps_out predicted bounding boxes.
    """
    n_features_out = encoder_input_test.shape[2]  
    num_test_samples = encoder_input_test.shape[0]
    #random_trajectory_samples = np.random.randint(low = 0, high = num_test_samples, size = hyperparams["num_test_predictions"])
    #sorted_random_trajectory_samples = sorted(random_trajectory_samples)
    source_trajectories = np.array(([1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40], [1023, 67, 26, 40]))
    predicted_trajectories = []
    print("Source trajectory # : \n {}".format(source_trajectories))
    src_traj_batched = np.expand_dims(source_trajectories, axis = 0)
    preds = predict_sequence(encoder_model, decoder_model, source_trajectory = src_traj_batched, n_steps_out = n_steps_out, n_features_out = n_features_out)
    preds = preds.astype("uint16")
    predicted_trajectories.append(preds)
    print("Predicted trajectory # : \n {}".format(preds))
    return predicted_trajectories

def main(args):
    """
    Main function
    """
    IN_HDF5_PATH = args.dataset
    saved_models_base_path = args.output
    config_file_path = args.config
    encoder_input_test, decoder_input_test, decoder_target_test = load_datasets(IN_HDF5_PATH, partition = 'test')
    max_output_trajectory_length = decoder_input_test.shape[1]
    train_model_path   = os.path.join(saved_models_base_path, "full_model.h5")
    encoder_model_path = os.path.join(saved_models_base_path, "encoder_model.h5")
    decoder_model_path = os.path.join(saved_models_base_path, "decoder_model.h5")
    model_history_path = os.path.join(saved_models_base_path, "model_history.json")
    full_model, encoder_model, decoder_model, train_history, hyperparams = load_models(train_model_path, encoder_model_path, decoder_model_path, model_history_path, config_file_path)
    #test_loss, test_metric = evaluate_models(full_model, encoder_input_test, decoder_input_test, decoder_target_test, hyperparams, train_history, saved_models_base_path)
    predicted_trajectories = path_predictor(encoder_model, decoder_model, hyperparams, encoder_input_test, decoder_target_test, max_output_trajectory_length)
    output_trajectories_namefile = "output_predictions"
    np.save(os.path.join(saved_models_base_path, output_trajectories_namefile), predicted_trajectories)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser(description ="CLI for making Path Predictions using trained models")
    parser.add_argument("dataset", help =" Path to input test dataset in .hdf5 format")
    parser.add_argument("output", help ="Path to the output directory where resulting training graphs and prediction results are computed.")
    parser.add_argument("config", help ="Path to configuration file used for testing.")
    args = parser.parse_args()
    main(args)