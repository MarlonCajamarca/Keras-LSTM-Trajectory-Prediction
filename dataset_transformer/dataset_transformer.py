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
import subprocess
import os

sys.path.append('..')


class SequenceXtractor(object):
	"""Sequence Extractor class"""
	def __init__(self, in_raw_data, hyperparams):
		self.in_raw_data = in_raw_data
		self.classes2extract = set(hyperparams["classes2extract"])
		self.trajectory_length = hyperparams["trajectory_length"]
		self.slide_window = hyperparams["slide_wdn"]
		self.numpy_data = []
	
	def append_trajectories(self, dataset, value):
		if value in self.classes2extract and len(dataset) >= self.trajectory_length:
			data_size = len(dataset)
			for slice_idx in range(0, data_size - self.trajectory_length + 1, self.slide_window):
				data_slice = dataset[slice_idx: slice_idx + self.trajectory_length]
				self.numpy_data.append(data_slice.astype("uint16"))
			if len(self.numpy_data) % 10000 == 0:
				print("Number of extracted trajectories: {}".format(len(self.numpy_data)))
	
	def get_hdf5_data(self, _, dataset):
		for value in dataset.attrs.values():
			self.append_trajectories(dataset, value)
	
	def extract(self):
		print("Running SequenceXtractor...")
		with h5py.File(self.in_raw_data, 'r+') as input_raw_data:
			start = time.time()
			input_raw_data.visititems(self.get_hdf5_data)
			np.random.shuffle(self.numpy_data)
			end = time.time()
			print(" Dataset successfully shuffled and processed in {} seconds!!!".format(end - start))
			return self.numpy_data


def lstm_dataset_generator(numpy_dataset, hyperparams):
	"""
	Creates input and target data for LSTM models
	------
	"""
	trajectory_stride = hyperparams["trajectory_stride"]
	max_in_trajectory_length = hyperparams["max_in_seq_length"]
	features_shape_0 = numpy_dataset.shape[0]
	features_shape_1 = numpy_dataset.shape[2]
	features_shape_2 = numpy_dataset.shape[1]
	start_gen = time.time()
	feature_trajectories = np.memmap("feature_trajectories", dtype="uint16", mode="w+", shape=(features_shape_0, features_shape_1, features_shape_2))
	print("Generating LSTM feature trajectories ....")
	for i, seq in enumerate(numpy_dataset):
		x_seq = np.array([bbox[0] for bbox in seq])
		y_seq = np.array([bbox[1] for bbox in seq])
		w_seq = np.array([bbox[2] for bbox in seq])
		h_seq = np.array([bbox[3] for bbox in seq])
		feature_trajectories[i][:][:] = [x_seq, y_seq, w_seq, h_seq]
		if i % 100000 == 0:
			feature_trajectories.flush()
			print("--- Flushed sequences : {}".format(i))
	feature_trajectories.flush()
	feature_trajectories = np.memmap("feature_trajectories", dtype="uint16", mode="r", shape=(features_shape_0, features_shape_1, features_shape_2))
	print(" Feature trajectories extracted!!!")
	print(" Example trajectory  :  {}".format(feature_trajectories[0]))
	lstm_input_data = feature_trajectories[:, :, :max_in_trajectory_length]
	lstm_target_data = feature_trajectories[:, :, max_in_trajectory_length::trajectory_stride]
	del_args = ('rm', "./feature_trajectories*")
	subprocess.call('%s %s' % del_args, shell=True)
	end_gen = time.time()
	print("  Dataset successfully generated in {} seconds!!!".format(end_gen - start_gen))
	print("  lstm_input_data: shape : {} , type : {}".format(lstm_input_data.shape, lstm_input_data.dtype))
	print("  lstm_target_data: shape : {} , type : {}".format(lstm_target_data.shape, lstm_target_data.dtype))
	return lstm_input_data, lstm_target_data


def lstm_target_sequence_normalizer(lstm_target_data, hyperparams):
	"""
	Smooths all the target trajectories using a polynomial interpolation of degree n
	"""
	polynomial_degree = hyperparams["polynomial_degree"]
	normalizer_axis = np.arange(lstm_target_data.shape[-1])
	smoothed_lstm_target_data = []
	start_norm = time.time()
	for i, raw_target_seq in enumerate(lstm_target_data):
		p_x = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[0], polynomial_degree))
		p_y = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[1], polynomial_degree))
		p_w = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[2], polynomial_degree))
		p_h = np.poly1d(np.polyfit(normalizer_axis, raw_target_seq[3], polynomial_degree))
		x_target_smoothed = np.array([int(p_x(t)) for t in normalizer_axis])
		y_target_smoothed = np.array([int(p_y(t)) for t in normalizer_axis])
		w_target_smoothed = np.array([int(p_w(t)) for t in normalizer_axis])
		h_target_smoothed = np.array([int(p_h(t)) for t in normalizer_axis])
		smoothed_lstm_target_data.append([x_target_smoothed, y_target_smoothed, w_target_smoothed, h_target_smoothed])
		if i % 100000 == 0:
			print(" Raw x_target : \n", raw_target_seq[0])
			print(" Smoothed x_target : \n", x_target_smoothed)
			print(" Raw y_target : \n", raw_target_seq[1])
			print(" Smoothed y_target : \n", y_target_smoothed)
			print(" Raw w_target : \n", raw_target_seq[2])
			print(" Smoothed w_target : \n", w_target_smoothed)
			print(" Raw h_target : \n", raw_target_seq[3])
			print(" Smoothed h_target : \n", h_target_smoothed)
			print(" Number of smoothed target trajectories : ", i)
	smoothed_lstm_target_data = np.array(smoothed_lstm_target_data).astype("uint16")
	end_norm = time.time()
	print("  Target trajectory data successfully normalized in {} seconds!!!".format(end_norm - start_norm))
	print("  Lstm smoothed target data: shape : {} , type : {}".format(smoothed_lstm_target_data.shape, smoothed_lstm_target_data.dtype))
	return smoothed_lstm_target_data


def train_test_splitter(lstm_input_data, lstm_target_data, split_idx):
	"""
	Splits the input and target LSTM data into train and test sets with a predefined partition size.
	"""
	x_train = lstm_input_data[:split_idx]
	y_train = lstm_target_data[:split_idx]
	x_test = lstm_input_data[split_idx:]
	y_test = lstm_target_data[split_idx:]
	print(" X_train shape: \n {}, type : {}".format(x_train.shape, x_train.dtype))
	print(" y_train shape: \n {}, type : {}".format(y_train.shape, y_train.dtype))
	print(" X_test shape: \n {}, type : {}".format(x_test.shape, x_test.dtype))
	print(" y_test shape: \n {}, type : {}".format(y_test.shape, y_test.dtype))
	return x_train, y_train, x_test, y_test


def print_trajectories(out_h5_path):
	"""
	Print trajectories from previously generated data
	"""
	out_x_train = HDF5Matrix(datapath=out_h5_path, dataset="train/X_train")
	out_y_train = HDF5Matrix(datapath=out_h5_path, dataset="train/y_train")
	num_test_samples = out_x_train.shape[0]
	random_trajectory_samples = np.random.randint(low=0, high=num_test_samples, size=5)
	sorted_random_trajectory_samples = sorted(random_trajectory_samples)
	source_trajectories = out_x_train[sorted_random_trajectory_samples]
	target_trajectories = out_y_train[sorted_random_trajectory_samples]
	for idx, source_trajectory in enumerate(source_trajectories):
		print("Source trajectory # {}: \n {}".format(idx, source_trajectory))
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
	if args.norm_target_data:
		print(" --> Normalizing target sequences...")
		lstm_target_data = lstm_target_sequence_normalizer(lstm_target_data, hyperparams)
	split_percentage = hyperparams["split_percentage"]
	split_idx = int(np.ceil(split_percentage * len(lstm_input_data)))
	x_train, y_train, x_test, y_test = train_test_splitter(lstm_input_data, lstm_target_data, split_idx)
	with h5py.File(out_h5_path, "a") as dataset:
		train_data = dataset.create_group("train")
		train_data.create_dataset("X_train", data=x_train)
		train_data.create_dataset("y_train", data=y_train)
		test_data = dataset.create_group("test")
		test_data.create_dataset("X_test", data=x_test)
		test_data.create_dataset("y_test", data=y_test)
		print("Train and Test Datasets already saved at {}".format(out_h5_path))
	print("---------- INFERENCE EXAMPLES -------------")
	print_trajectories(out_h5_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("raw_dataset", help=" Path to input raw dataset in .h5 format")
	parser.add_argument("out_dataset", help=" Path to output train/test dataset in .h5 format")
	parser.add_argument("config", help="Path to configuration file used for dataset transformer tool.")
	parser.add_argument("--norm_target_data", action="store_true", default=False)
	args = parser.parse_args()
	main(args)
