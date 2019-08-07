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

sys.path.append('..')

class SequenceXtractor(object):
    """ Blueprint for sequence extractor object """
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
            if (len(self.numpy_dataset) % 1000 == 0):
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
    sos_token = hyperparams["sos_token"]
    traj_len = hyperparams["traj_len"]
    max_in_trajectory_length = hyperparams["max_in_seq_length"]
    feature_trajectories = []
    for seq in numpy_dataset:
        x_seq = np.array([bbox[0] for bbox in seq])
        y_seq = np.array([bbox[1] for bbox in seq])
        w_seq = np.array([bbox[2] for bbox in seq])
        h_seq = np.array([bbox[3] for bbox in seq])
        feature_trajectories.append([x_seq, y_seq, w_seq, h_seq])
    feature_trajectories = np.array(feature_trajectories)
    print(" Feature trajectories extracted!!!")
    lstm_input_data = feature_trajectories[:,:,:max_in_trajectory_length]
    lstm_target_data = feature_trajectories[:,:,max_in_trajectory_length:]
    print("  lstm_input_data: shape : {} , type : {}".format(lstm_input_data.shape, lstm_input_data.dtype))
    print("  lstm_target_data: shape : {} , type : {}".format(lstm_target_data.shape, lstm_target_data.dtype))
    return lstm_input_data, lstm_target_data

def train_test_splitter(lstm_input_data, lstm_target_data, split_idx):
    X_train = lstm_input_data[:split_idx]
    y_train = lstm_target_data[:split_idx]
    X_test = lstm_input_data[split_idx:]
    y_test = lstm_target_data[split_idx:]
    print(" X_train shape: \n {}, type : {}".format(X_train.shape, X_train.dtype))
    print(" y_train shape: \n {}, type : {}".format(y_train.shape, y_train.dtype))
    print(" X_test shape: \n {}, type : {}".format(X_test.shape, X_test.dtype))
    print(" y_test shape: \n {}, type : {}".format(y_test.shape, y_test.dtype))
    return  X_train, y_train, X_test, y_test

def main(args):
    print("Dataset Transformer tool has started!")
    in_h5_path = args.raw_dataset
    out_h5_path = args.out_dataset
    config_file_path = args.config
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)
    seq_xtractor = SequenceXtractor(in_h5_path, hyperparams)
    numpy_dataset = seq_xtractor.extract()
    lstm_input_data, lstm_target_data = lstm_dataset_generator(numpy_dataset, hyperparams)
    split_percentage = hyperparams["split_percentage"]
    split_idx = int(np.ceil(split_percentage * len(lstm_input_data)))
    X_train, y_train, X_test, y_test = train_test_splitter(lstm_input_data, lstm_target_data, split_idx)
    with h5py.File(out_h5_path, "a") as dataset:
        train_data = dataset.create_group("train")
        train_data.create_dataset("X_train", data = X_train)
        train_data.create_dataset("y_train", data = y_train)
        test_data =dataset.create_group("test")
        test_data.create_dataset("X_test", data = X_test)
        test_data.create_dataset("y_test", data = y_test)
        print("Train and Test Datasets already saved at {}".format(out_h5_path))
    #  Dubugging output dataset
    out_X_train = HDF5Matrix(datapath=out_h5_path, dataset = "train/X_train")
    out_y_train = HDF5Matrix(datapath=out_h5_path, dataset = "train/y_train")
    n_features_out = out_X_train.shape[2]  
    num_test_samples = out_X_train.shape[0]
    random_trajectory_samples = np.random.randint(low = 0, high = num_test_samples, size = hyperparams["num_test_predictions"])
    sorted_random_trajectory_samples = sorted(random_trajectory_samples)
    source_trajectories = out_X_train[sorted_random_trajectory_samples]
    target_trajectories = out_y_train[sorted_random_trajectory_samples]
    for idx, src_traj in enumerate(source_trajectories):
        print("Source trajectory # {}: \n {}".format(idx, src_traj))
        print("Target trajectory # {}: \n {}".format(idx, target_trajectories[idx]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset", help =" Path to input raw dataset in .h5 format")
    parser.add_argument("out_dataset", help =" Path to output train/test dataset in .h5 format")
    parser.add_argument("config", help ="Path to configuration file used for dataset transformer tool.")
    args = parser.parse_args()
    main(args)