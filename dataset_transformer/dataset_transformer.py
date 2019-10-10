#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marlon Andres Cajamarca

Path Prediction LSTM - Dataset Transformator tool for generating train and test sets for LSTM models
"""
import numpy as np
import h5py
import json
import time
import sys
import argparse
from keras.utils import HDF5Matrix
import os

sys.path.append('..')

class SequenceXtractor(object):
    """Sequence Extractor class"""
    def __init__(self, in_raw_dataset, hyperparams):
        self.in_raw_dataset = in_raw_dataset
        self.classes2extract = set(hyperparams["classes2extract"])
        self.traj_len = hyperparams["traj_len"]
        self.slide_window = hyperparams["slide_wdn"]
        self.numpy_dataset = []   
    def append_trajectories(self, dataset, value):
        if (value in self.classes2extract and len(dataset) >= self.traj_len):
            dataset_size = len(dataset)
            for slice_idx in range(0, dataset_size - self.traj_len + 1, self.slide_window):
                dataset_slice = dataset[slice_idx : slice_idx + self.traj_len]
                self.numpy_dataset.append(dataset_slice.astype("uint16"))
            if (len(self.numpy_dataset) % 10000 == 0):
                print("Number of extracted trajectories: {}".format(len(self.numpy_dataset))) 
    def get_hdf5_data(self, name, dataset):
        for value in dataset.attrs.values():
            self.append_trajectories(dataset, value)
    def extract(self):
        print("Running SequenceXtractor...")
        with h5py.File(self.in_raw_dataset, 'r+') as input_raw_dataset:
            start = time.time()
            input_raw_dataset.visititems(self.get_hdf5_data)
            np.random.shuffle(self.numpy_dataset)
            end = time.time()
            print(" Dataset successfully shuffled and processed in {} seconds!!!".format(end - start))
            return self.numpy_dataset

def lstm_dataset_generator(numpy_dataset, hyperparams):
    """
    Creates input and target data for LSTM models
    
    Parameters
    ----------
    numpy_dataset -> str : Raw numpy dataset stored in RAM and containing all extracted sequences
    hyperparams -> str : Parameters to control how the output tensors will be generated
    
    Returns :
    lstm_input_data -> Tensor : Tensor containing input trajectories for LSTM model
    lstm_target_data -> Tensor : Tensor containing target trajectories for LSTM model
    ------
    """
    traj_stride = hyperparams["traj_stride"]
    max_in_trajectory_length = hyperparams["max_in_seq_length"]
    features_shape_0 = numpy_dataset.shape[0]
    features_shape_1 = numpy_dataset.shape[2]
    features_shape_2 = numpy_dataset.shape[1]
    start_gen = time.time()
    # Saving extracted trajectories into a memory mapped array saved on disk to avoid RAM memory issues
    feature_trajectories = np.memmap("feature_trajectories", dtype = "uint16", mode = "w+", shape = (features_shape_0, features_shape_1, features_shape_2))
    print("Generating LSTM feature trajectories ....")
    # Flushing the secuences into the memmap array
    for i, seq in enumerate(numpy_dataset):
        x_seq = np.array([bbox[0] for bbox in seq])
        y_seq = np.array([bbox[1] for bbox in seq])
        w_seq = np.array([bbox[2] for bbox in seq])
        h_seq = np.array([bbox[3] for bbox in seq])
        feature_trajectories[i][:][:] = [x_seq, y_seq, w_seq, h_seq]
        # Flush the trajectories into disk every 100000 trajectories
        if (i % 100000 == 0):
            feature_trajectories.flush()
            print("--- Flushed sequences : {}".format(i))
    feature_trajectories.flush()
    # emptying the original dataset to recover RAM memmory space
    numpy_dataset = []
    numpy_dataset = None
    # Reading from memmap array to create input and target LSTM data 
    feature_trajectories = np.memmap("feature_trajectories", dtype = "uint16", mode = "r", shape =(features_shape_0, features_shape_1, features_shape_2)) 
    print(" Feature trajectories extracted!!!")
    print(" Example trajectory  :  {}".format(feature_trajectories[0]))
    lstm_input_data = feature_trajectories[:,:,:max_in_trajectory_length]
    lstm_target_data = feature_trajectories[:,:,max_in_trajectory_length::traj_stride]
    # Deleting the memmap array 
    del feature_trajectories
    end_gen = time.time()
    print("  Dataset successfully generated in {} seconds!!!".format(end_gen - start_gen))
    print("  lstm_input_data: shape : {} , type : {}".format(lstm_input_data.shape, lstm_input_data.dtype))
    print("  lstm_target_data: shape : {} , type : {}".format(lstm_target_data.shape, lstm_target_data.dtype))
    return lstm_input_data, lstm_target_data

def lstm_target_sequence_normalizer(lstm_target_data, hyperparams):
    """
    Smothes all the target trajectories using a polinomial interpolation of degree n
    
    Parameters
    ----------
    lstm_target_data -> Tensor : Un-smoothed LSTM target trajectories
    
    Returns :
    smoothed_lstm_target_data -> Tensor : Smoothed LSTM target trajectories
    ------
    """
    polinomial_degree = hyperparams["polinomial_degree"]
    normalizer_axis = np.arange(lstm_target_data.shape[-1])
    smoothed_lstm_target_data = []
    start_norm = time.time()
    for i, raw_target_seq in enumerate(lstm_target_data):
        # Calculating 3rd degree polynomial coeeficients for curve fitting
        p_x = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[0], polinomial_degree))
        p_y = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[1], polinomial_degree))
        p_w = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[2], polinomial_degree))
        p_h = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[3], polinomial_degree))
        # Calculating smoothed target trayectories from corresponding polinomial reggresion coefficients
        x_target_smoothed = np.array([int(p_x(t)) for t in normalizer_axis])
        y_target_smoothed = np.array([int(p_y(t)) for t in normalizer_axis])
        w_target_smoothed = np.array([int(p_w(t)) for t in normalizer_axis])
        h_target_smoothed = np.array([int(p_h(t)) for t in normalizer_axis])
        smoothed_lstm_target_data.append([x_target_smoothed, y_target_smoothed, w_target_smoothed, h_target_smoothed])
        # Creating the smoothed lstm target data
        if (i % 100000 == 0):
            print(" Raw x_target : \n", raw_target_seq[0])
            print(" Smoothed x_target : \n", x_target_smoothed)
            print(" Raw y_target : \n", raw_target_seq[1])
            print(" Smoothed y_target : \n", y_target_smoothed)
            print(" Raw w_target : \n", raw_target_seq[2])
            print(" Smoothed w_target : \n", w_target_smoothed)
            print(" Raw h_target : \n", raw_target_seq[3])
            print(" Smoothed h_target : \n", h_target_smoothed)
            print(" Number of smoothed target trajectories : ", i)
    lstm_target_data = None
    smoothed_lstm_target_data = np.array(smoothed_lstm_target_data).astype("uint16")
    end_norm = time.time()
    print("  Dataset successfully normalized in {} seconds!!!".format(end_norm - start_norm))
    print("  Lstm smoothed target data: shape : {} , type : {}".format(smoothed_lstm_target_data.shape, smoothed_lstm_target_data.dtype))

    return smoothed_lstm_target_data

def train_test_splitter(lstm_input_data, lstm_target_data, split_idx):
    """
    Splits the input and target LSTM data into train and test sets with a predefined partition size.
    
    Parameters
    ----------
    lstm_input_data -> Tensor : Full input LSTM data tensor
    lstm_target_data -> Tensor : Full output LSTM data tensor
    split_idx -> float : Input idx position to make the splitting process
    
    Returns :
    X_train -> Tensor : Input train trajectories to train LSTM trajectory prediction models
    Y_train -> Tensor : Target train trajectories to train LSTM trajectory prediction models
    X_test -> Tensor : Input test trajectories to evaluate LSTM trajectory prediction models
    Y_test -> Tensor : Target test trajectories to evaluate LSTM trajectory prediction models
    ------
    """
    X_train = lstm_input_data[:split_idx]
    y_train = lstm_target_data[:split_idx]
    X_test = lstm_input_data[split_idx:]
    y_test = lstm_target_data[split_idx:]
    # Deleting unused tensors in order to free memory
    lstm_target_data = None
    lstm_input_data = None
    print(" X_train shape: \n {}, type : {}".format(X_train.shape, X_train.dtype))
    print(" y_train shape: \n {}, type : {}".format(y_train.shape, y_train.dtype))
    print(" X_test shape: \n {}, type : {}".format(X_test.shape, X_test.dtype))
    print(" y_test shape: \n {}, type : {}".format(y_test.shape, y_test.dtype))

    return  X_train, y_train, X_test, y_test

def print_trajectories(out_h5_path):
    """
    Print trajectories from previously generated datasets
    
    Parameters
    ----------
    out_X_train -> str : Path to the first dataset to merge
    out_y_train -> str : Path to the second dataset to merge
    
    Returns : void
    ------
    """
    out_X_train = HDF5Matrix(datapath=out_h5_path, dataset = "train/X_train")
    out_y_train = HDF5Matrix(datapath=out_h5_path, dataset = "train/y_train")
    n_features_out = out_X_train.shape[2]  
    num_test_samples = out_X_train.shape[0]
    random_trajectory_samples = np.random.randint(low = 0, high = num_test_samples, size = 5)
    sorted_random_trajectory_samples = sorted(random_trajectory_samples)
    source_trajectories = out_X_train[sorted_random_trajectory_samples]
    target_trajectories = out_y_train[sorted_random_trajectory_samples]
    for idx, src_traj in enumerate(source_trajectories):
        print("Source trajectory # {}: \n {}".format(idx, src_traj))
        print("Target trajectory # {}: \n {}".format(idx, target_trajectories[idx]))

def main(args):
    """
    Main Function
    """
    print("Dataset Transformer tool has started!")
    in_h5_path = args.raw_dataset
    out_h5_path = args.out_dataset
    config_file_path = args.config
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)
    seq_xtractor = SequenceXtractor(in_h5_path, hyperparams)
    numpy_dataset = seq_xtractor.extract()
    numpy_dataset = np.array(numpy_dataset)
    lstm_input_data, lstm_target_data = lstm_dataset_generator(numpy_dataset, hyperparams)
    smoothed_lstm_target_data = lstm_target_sequence_normalizer(lstm_target_data, hyperparams)
    split_percentage = hyperparams["split_percentage"]
    split_idx = int(np.ceil(split_percentage * len(lstm_input_data)))
    X_train, y_train, X_test, y_test = train_test_splitter(lstm_input_data, smoothed_lstm_target_data, split_idx)
    # Creating the output dataset with previously generated train and test data
    with h5py.File(out_h5_path, "a") as dataset:
        train_data = dataset.create_group("train")
        train_data.create_dataset("X_train", data = X_train)
        train_data.create_dataset("y_train", data = y_train)
        test_data =dataset.create_group("test")
        test_data.create_dataset("X_test", data = X_test)
        test_data.create_dataset("y_test", data = y_test)
        print("Train and Test Datasets already saved at {}".format(out_h5_path))
    #  Dubugging output dataset
    print("---------- INFERENCE EXAMPLES -------------")
    print_trajectories(out_h5_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset", help =" Path to input raw dataset in .h5 format")
    parser.add_argument("out_dataset", help =" Path to output train/test dataset in .h5 format")
    parser.add_argument("config", help ="Path to configuration file used for dataset transformer tool.")
    args = parser.parse_args()
    main(args)