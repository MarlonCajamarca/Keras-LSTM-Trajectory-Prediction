#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marlon Andres Cajamarca

Dataset Merger Utility tool: Merge two already created train/test .hdf5 daatsets into a single output dataset.
"""

import h5py
import numpy as np
from keras.utils import HDF5Matrix
import time
import argparse
import random

_DEFAULT_X_TRAIN_PARTITION = "train/X_train"
_DEFAULT_Y_TRAIN_PARTITION = "train/y_train"
_DEFAULT_X_TEST_PARTITION = "test/X_test"
_DEFAULT_Y_TEST_PARTITION = "test/y_test"
_DEFAULT_SHUFFLE_SEED = 19


class DatasetMerger(object):
	def __init__(self, dataset_1_path, dataset_2_path, out_h5_path):
		self.dataset_1_x_train = HDF5Matrix(datapath=dataset_1_path, dataset=_DEFAULT_X_TRAIN_PARTITION)
		self.dataset_1_y_train = HDF5Matrix(datapath=dataset_1_path, dataset=_DEFAULT_Y_TRAIN_PARTITION)
		self.dataset_1_x_test = HDF5Matrix(datapath=dataset_1_path, dataset=_DEFAULT_X_TEST_PARTITION)
		self.dataset_1_y_test = HDF5Matrix(datapath=dataset_1_path, dataset=_DEFAULT_Y_TEST_PARTITION)
		self.dataset_2_x_train = HDF5Matrix(datapath=dataset_2_path, dataset=_DEFAULT_X_TRAIN_PARTITION)
		self.dataset_2_y_train = HDF5Matrix(datapath=dataset_2_path, dataset=_DEFAULT_Y_TRAIN_PARTITION)
		self.dataset_2_x_test = HDF5Matrix(datapath=dataset_2_path, dataset=_DEFAULT_X_TEST_PARTITION)
		self.dataset_2_y_test = HDF5Matrix(datapath=dataset_2_path, dataset=_DEFAULT_Y_TEST_PARTITION)
		self.full_x_train_dataset = None
		self.full_y_train_dataset = None
		self.full_x_test_dataset = None
		self.full_y_test_dataset = None
		self.out_hdf5_path = out_h5_path

	def run(self):
		start = time.time()
		self.shuffle_and_merge_data()
		self.generate_output_merged_dataset()
		self.print_debug_trajectories()
		end = time.time()
		print(" Datasets successfully merged in {} seconds!!!".format(end - start))

	def shuffle_and_merge_data(self):
		full_x_train_dataset_raw = np.concatenate((self.dataset_1_x_train[:], self.dataset_2_x_train[:]), axis=0)
		full_y_train_dataset_raw = np.concatenate((self.dataset_1_y_train[:], self.dataset_2_y_train[:]), axis=0)
		full_x_test_dataset_raw = np.concatenate((self.dataset_1_x_test[:], self.dataset_2_x_test[:]), axis=0)
		full_y_test_dataset_raw = np.concatenate((self.dataset_1_y_test[:], self.dataset_2_y_test[:]), axis=0)
		full_train_dataset_shuffle = list(zip(full_x_train_dataset_raw, full_y_train_dataset_raw))
		full_test_dataset_shuffle = list(zip(full_x_test_dataset_raw, full_y_test_dataset_raw))
		random.seed(_DEFAULT_SHUFFLE_SEED)
		np.random.shuffle(full_train_dataset_shuffle)
		np.random.shuffle(full_test_dataset_shuffle)
		self.full_x_train_dataset, self.full_y_train_dataset = map(list, zip(*full_train_dataset_shuffle))
		self.full_x_test_dataset, self.full_y_test_dataset = map(list, zip(*full_test_dataset_shuffle))
		self.full_x_train_dataset = np.array(self.full_x_train_dataset)
		self.full_y_train_dataset = np.array(self.full_y_train_dataset)
		self.full_x_test_dataset = np.array(self.full_x_test_dataset)
		self.full_y_test_dataset = np.array(self.full_y_test_dataset)
		print(" ---------- Datasets Merging Results ----------")
		print(" --> Full_X_train_dataset shape: \n {}, type : {}".format(self.full_x_train_dataset.shape, self.full_x_train_dataset.dtype))
		print(" 1_X_train_dataset shape: \n {}, type : {}".format(self.dataset_1_x_train[:].shape, self.dataset_1_x_train[:].dtype))
		print(" 2_X_train_dataset shape: \n {}, type : {}".format(self.dataset_2_x_train[:].shape, self.dataset_2_x_train[:].dtype))
		print(" --> Full_Y_train_dataset shape: \n {}, type : {}".format(self.full_y_train_dataset.shape, self.full_y_train_dataset.dtype))
		print(" 1_Y_train_dataset shape: \n {}, type : {}".format(self.dataset_1_y_train[:].shape, self.dataset_1_y_train[:].dtype))
		print(" 2_Y_train_dataset shape: \n {}, type : {}".format(self.dataset_2_y_train[:].shape, self.dataset_2_y_train[:].dtype))
		print(" --> Full_X_test_dataset shape: \n {}, type : {}".format(self.full_x_test_dataset.shape, self.full_x_test_dataset.dtype))
		print(" 1_X_test_dataset shape: \n {}, type : {}".format(self.dataset_1_x_test[:].shape, self.dataset_1_x_test[:].dtype))
		print(" 2_X_test_dataset shape: \n {}, type : {}".format(self.dataset_2_x_test[:].shape, self.dataset_2_x_test[:].dtype))
		print(" --> Full_Y_test_dataset shape: \n {}, type : {}".format(self.full_y_test_dataset.shape, self.full_y_test_dataset.dtype))
		print(" 1_Y_test_dataset shape: \n {}, type : {}".format(self.dataset_1_y_test[:].shape, self.dataset_1_y_test[:].dtype))
		print(" 2_Y_test_dataset shape: \n {}, type : {}".format(self.dataset_2_y_test[:].shape, self.dataset_2_y_test[:].dtype))

	def generate_output_merged_dataset(self):
		# Creating the output .hdf5 datasets with the previously generated data
		with h5py.File(self.out_hdf5_path, "a") as output_dataset:
			train_data = output_dataset.create_group("train")
			train_data.create_dataset("X_train", data=self.full_x_train_dataset)
			train_data.create_dataset("y_train", data=self.full_y_train_dataset)
			test_data = output_dataset.create_group("test")
			test_data.create_dataset("X_test", data=self.full_x_test_dataset)
			test_data.create_dataset("y_test", data=self.full_y_test_dataset)
			print("Train and Test Datasets already saved at {}".format(self.out_hdf5_path))
			print("Datasets successfully merged and saved!")

	def print_debug_trajectories(self):
		"""
	    Print trajectories from previously generated datasets
	    """
		out_x_test = HDF5Matrix(datapath=self.out_hdf5_path, dataset=_DEFAULT_X_TEST_PARTITION)
		out_y_test = HDF5Matrix(datapath=self.out_hdf5_path, dataset=_DEFAULT_Y_TEST_PARTITION)
		num_test_samples = out_x_test.shape[0]
		random_trajectory_samples = np.random.randint(low=0, high=num_test_samples, size=5)
		sorted_random_trajectory_samples = sorted(random_trajectory_samples)
		source_trajectories = out_x_test[sorted_random_trajectory_samples]
		target_trajectories = out_y_test[sorted_random_trajectory_samples]
		for idx, src_trajectory in enumerate(source_trajectories):
			print("Source trajectory # {}: \n {}".format(idx, src_trajectory))
			print("Target trajectory # {}: \n {}".format(idx, target_trajectories[idx]))


if __name__ == "__main__":
	""" This is executed when run from the command line """
	parser = argparse.ArgumentParser(description="CLI for merging .hdf5 datasets used for LSTM model's training and testing")
	parser.add_argument("input_dataset_1", help=" Full path to first input  dataset in .hdf5 format to merge")
	parser.add_argument("input_dataset_2", help=" Full path to second input dataset in .hdf5 format to merge")
	parser.add_argument("out_dataset", help=" Path to output train/test dataset in .h5 format")
	args = parser.parse_args()
	input_dataset_1_path = args.input_dataset_1
	input_dataset_2_path = args.input_dataset_2
	out_h5_path = args.out_dataset
	DatasetMerger(input_dataset_1_path, input_dataset_2_path, out_h5_path).run()
