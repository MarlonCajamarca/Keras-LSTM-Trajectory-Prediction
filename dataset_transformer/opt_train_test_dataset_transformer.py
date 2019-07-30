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

def enc_dec_dataset_generator(numpy_dataset, hyperparams):
    """ Generate the encoder and decoder input data used to create train and test sets
    
    """
    sos_token = hyperparams["sos_token"]
    traj_len = hyperparams["traj_len"]
    max_in_trajectory_length = hyperparams["max_in_seq_length"]
    max_out_trajectory_length = traj_len - max_in_trajectory_length - 1
    sos_idx_vector = np.array([sos_token, sos_token, sos_token, sos_token]).astype("uint16")
    
    encoder_input_data = []
    for seq in numpy_dataset:
        encoder_input_data.append(seq[:max_in_trajectory_length])
    encoder_input_data = np.array(encoder_input_data)
    
    decoder_input_data = []
    for seq in numpy_dataset:
        decoder_in_data = np.vstack((sos_idx_vector, seq[max_in_trajectory_length: max_in_trajectory_length + max_out_trajectory_length]))
        decoder_input_data.append(decoder_in_data)
    decoder_input_data = np.array(decoder_input_data)
    
    decoder_target_data = []
    for seq in numpy_dataset:
        decoder_tg_data = seq[max_in_trajectory_length: max_in_trajectory_length + max_out_trajectory_length + 1]
        decoder_target_data.append(decoder_tg_data)
    decoder_target_data = np.array(decoder_target_data)

    print("  Encoder Input tensor: shape : {} , type : {}".format(encoder_input_data.shape, encoder_input_data.dtype))
    print("  Decoder Input tensor: shape : {} , type : {}".format(decoder_input_data.shape, decoder_input_data.dtype))
    print("  Decoder Target tensor: shape : {} , type : {}".format(decoder_target_data.shape, decoder_target_data.dtype))
    return encoder_input_data, decoder_input_data, decoder_target_data

def train_test_splitter(encoder_input_data, decoder_input_data, decoder_target_data, split_idx):
    encoder_input_train = encoder_input_data[:split_idx]
    decoder_input_train = decoder_input_data[:split_idx]
    decoder_target_train = decoder_target_data[:split_idx]
    encoder_input_test = encoder_input_data[split_idx:]
    decoder_input_test = decoder_input_data[split_idx:]
    decoder_target_test = decoder_target_data[split_idx:]
    print(" Encoder input train shape: \n {}, type : {}".format(encoder_input_train.shape, encoder_input_train.dtype))
    print(" Decoder input train shape: \n {}, type : {}".format(decoder_input_train.shape, decoder_input_train.dtype))
    print(" Decoder target train shape: \n {}, type : {}".format(decoder_target_train.shape, decoder_target_train.dtype))
    print(" Encoder input test shape: \n {}, type : {}".format(encoder_input_test.shape, encoder_input_test.dtype))
    print(" Decoder input test shape: \n {}, type : {}".format(decoder_input_test.shape, decoder_input_test.dtype))
    print(" Decoder target test shape: \n {}, type : {}".format(decoder_target_test.shape, decoder_target_test.dtype))
    return  encoder_input_train, decoder_input_train, decoder_target_train, encoder_input_test, decoder_input_test, decoder_target_test

def main(args):
    print("Dataset Transformer tool Started!")
    in_h5_path = args.raw_dataset
    out_h5_path = args.out_dataset
    config_file_path = args.config
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)
    seq_xtractor = SequenceXtractor(in_h5_path, hyperparams)
    numpy_dataset = seq_xtractor.extract()
    encoder_input_data, decoder_input_data, decoder_target_data = enc_dec_dataset_generator(numpy_dataset, hyperparams)
    split_percentage = hyperparams["split_percentage"]
    split_idx = int(np.ceil(split_percentage * len(encoder_input_data)))
    encoder_input_train, decoder_input_train, decoder_target_train, encoder_input_test, decoder_input_test, decoder_target_test = train_test_splitter(encoder_input_data, decoder_input_data, decoder_target_data, split_idx)
    with h5py.File(out_h5_path, "a") as dataset:
        train_data = dataset.create_group("train")
        train_data.create_dataset("encoder_in", data = encoder_input_train)
        train_data.create_dataset("decoder_in", data = decoder_input_train)
        train_data.create_dataset("decoder_target", data = decoder_target_train)
        test_data =dataset.create_group("test")
        test_data.create_dataset("encoder_in", data = encoder_input_test)
        test_data.create_dataset("decoder_in", data = decoder_input_test)
        test_data.create_dataset("decoder_target", data = decoder_target_test)
        print("Train and Test Datasets already saved at {}".format(out_h5_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset", help =" Path to input raw dataset in .h5 format")
    parser.add_argument("out_dataset", help =" Path to output train/test dataset in .h5 format")
    parser.add_argument("config", help ="Path to configuration file used for dataset transformer tool.")
    args = parser.parse_args()
    main(args)