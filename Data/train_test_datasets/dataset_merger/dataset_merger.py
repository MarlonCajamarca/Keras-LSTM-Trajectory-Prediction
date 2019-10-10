'''
HDF5 training test dataset merger tool

'''

import h5py
import numpy as np
from keras.utils import HDF5Matrix
import time
import os
import sys
import argparse
import random

def merge_datasets(dataset_1_path, dataset_2_path, out_h5_path):
    """
    Load previuously trained model for evaluation and inference purposes
    
    Parameters
    ----------
    dataset_1_path -> str : Path to the first dataset to merge
    dataset_2_path -> str : Path to the second dataset to merge
    out_h5_path -> str : Path to the output .hdf5 dataset containing merged datasets
    
    Returns : void
    ------
    """
    # Loading first dataset train and test sets
    dataset_1_X_train = HDF5Matrix(datapath=dataset_1_path, dataset = "train/X_train")
    dataset_1_Y_train = HDF5Matrix(datapath=dataset_1_path, dataset = "train/y_train")
    dataset_1_X_test = HDF5Matrix(datapath=dataset_1_path, dataset = "test/X_test")
    dataset_1_Y_test = HDF5Matrix(datapath=dataset_1_path, dataset = "test/y_test")
    # Loading second dataset train and test sets
    dataset_2_X_train = HDF5Matrix(datapath=dataset_2_path, dataset = "train/X_train")
    dataset_2_Y_train = HDF5Matrix(datapath=dataset_2_path, dataset = "train/y_train")
    dataset_2_X_test = HDF5Matrix(datapath=dataset_2_path, dataset = "test/X_test")
    dataset_2_Y_test = HDF5Matrix(datapath=dataset_2_path, dataset = "test/y_test")
    #Specifying shuffle seed value to make the shuffling process reproducible
    shuffle_seed = 19
    # Merging corresponding X and y data from input datasets
    full_X_train_dataset_raw = np.concatenate((dataset_1_X_train[:], dataset_2_X_train[:]), axis=0)
    full_Y_train_dataset_raw = np.concatenate((dataset_1_Y_train[:], dataset_2_Y_train[:]), axis=0)
    full_X_test_dataset_raw = np.concatenate((dataset_1_X_test[:], dataset_2_X_test[:]), axis=0)
    full_Y_test_dataset_raw = np.concatenate((dataset_1_Y_test[:], dataset_2_Y_test[:]), axis=0)
    # Zipping the train and test partitions to shuffle X and y accordingly
    full_train_dataset_shuffle = list(zip(full_X_train_dataset_raw, full_Y_train_dataset_raw))
    full_test_dataset_shuffle = list(zip(full_X_test_dataset_raw, full_Y_test_dataset_raw))
    # Make the shuffle of train and test partitions
    random.seed(shuffle_seed)
    np.random.shuffle(full_train_dataset_shuffle)
    np.random.shuffle(full_test_dataset_shuffle)
    # Unpacking the shuffled train and test partitions
    full_X_train_dataset, full_Y_train_dataset = map(list, zip(*full_train_dataset_shuffle))
    full_X_test_dataset, full_Y_test_dataset = map(list, zip(*full_test_dataset_shuffle))
    # From tuple to np.array to acces size and type
    full_X_train_dataset = np.array(full_X_train_dataset)
    full_Y_train_dataset = np.array(full_Y_train_dataset)
    full_X_test_dataset = np.array(full_X_test_dataset)
    full_Y_test_dataset = np.array(full_Y_test_dataset)
    # Printing results + Debbuging
    print(" ---------- Datasets Merging Results ----------")
    print(" full_X_train_dataset shape: \n {}, type : {}".format(full_X_train_dataset.shape, full_X_train_dataset.dtype))
    print(" 1_X_train_dataset shape: \n {}, type : {}".format(dataset_1_X_train[:].shape, dataset_1_X_train[:].dtype))
    print(" 2_X_train_dataset shape: \n {}, type : {}".format(dataset_2_X_train[:].shape, dataset_2_X_train[:].dtype))

    print(" full_Y_train_dataset shape: \n {}, type : {}".format(full_Y_train_dataset.shape, full_Y_train_dataset.dtype))
    print(" 1_Y_train_dataset shape: \n {}, type : {}".format(dataset_1_Y_train[:].shape, dataset_1_Y_train[:].dtype))
    print(" 2_Y_train_dataset shape: \n {}, type : {}".format(dataset_2_Y_train[:].shape, dataset_2_Y_train[:].dtype))

    print(" full_X_test_dataset shape: \n {}, type : {}".format(full_X_test_dataset.shape, full_X_test_dataset.dtype))
    print(" 1_X_test_dataset shape: \n {}, type : {}".format(dataset_1_X_test[:].shape, dataset_1_X_test[:].dtype))
    print(" 2_X_test_dataset shape: \n {}, type : {}".format(dataset_2_X_test[:].shape, dataset_2_X_test[:].dtype))

    print(" full_Y_test_dataset shape: \n {}, type : {}".format(full_Y_test_dataset.shape, full_Y_test_dataset.dtype))
    print(" 1_Y_test_dataset shape: \n {}, type : {}".format(dataset_1_Y_test[:].shape, dataset_1_Y_test[:].dtype))
    print(" 2_Y_test_dataset shape: \n {}, type : {}".format(dataset_2_Y_test[:].shape, dataset_2_Y_test[:].dtype))
    print(" ----------------------------------------------")
   	# Creating the output .hdf5 datasets with the previously generated data
    with h5py.File(out_h5_path, "a") as output_dataset:
	    train_data = output_dataset.create_group("train")
	    train_data.create_dataset("X_train", data = full_X_train_dataset)
	    train_data.create_dataset("y_train", data = full_Y_train_dataset)
	    test_data =output_dataset.create_group("test")
	    test_data.create_dataset("X_test", data = full_X_test_dataset)
	    test_data.create_dataset("y_test", data = full_Y_test_dataset)
	    print("Train and Test Datasets already saved at {}".format(out_h5_path))
	    print("Datasets successfully merged!")

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
    Main function
    """
    start = time.time()
    input_dataset_1_path = args.input_dataset_1
    input_dataset_2_path = args.input_dataset_2
    out_h5_path = args.out_dataset
    merge_datasets(input_dataset_1_path, input_dataset_2_path, out_h5_path)
    # Loading the recently merged dataset to inspect the trajectory samples
    print_trajectories(out_h5_path)
    end = time.time()
    print(" Datasets successfully merged in {} seconds!!!".format(end - start))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser(description ="CLI for merging .hdf5 datasets used for LSTM model's training and testing")
    parser.add_argument("input_dataset_1", help =" Full path to first input  dataset in .hdf5 format to merge")
    parser.add_argument("input_dataset_2", help =" Full path to second input dataset in .hdf5 format to merge")
    parser.add_argument("out_dataset", help =" Path to output train/test dataset in .h5 format")
    args = parser.parse_args()
    main(args)