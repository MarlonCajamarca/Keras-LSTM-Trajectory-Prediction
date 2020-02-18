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
from keras.utils import HDF5Matrix
from keras import backend
from keras.models import load_model
from keras_radam import RAdam
from keras_lookahead import Lookahead


def load_datasets(in_h5_path, partition='train'):
	"""
    Load train or test dataset
    """
	if partition == 'train':
		x_train = HDF5Matrix(datapath=in_h5_path, dataset="train/X_train")
		y_train = HDF5Matrix(datapath=in_h5_path, dataset="train/y_train")
		return x_train, y_train
	elif partition == 'test':
		x_test = HDF5Matrix(datapath=in_h5_path, dataset="test/X_test")
		y_test = HDF5Matrix(datapath=in_h5_path, dataset="test/y_test")
		return x_test, y_test
	else:
		print("Invalid 'partition' parameter: Valid values: ['train', 'test']")


def rmse(y_true, y_pred):
	"""
    Calculate Root Mean Squared Error custom metric
    """
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def load_models(lstm_model_path, model_history_path, config_file_path):
	"""
    Load previously trained model for evaluation and inference purposes
    """
	lstm_model = load_model(lstm_model_path, custom_objects={'rmse': rmse, 'Lookahead': Lookahead, 'RAdam': RAdam})
	lstm_model.summary(line_length=180)
	# Printing the input and output model's layer names. They will be needed in the Classifier to load the LSTM models correctly.
	print("-----_model_inputs------")
	print(lstm_model.inputs)
	print("----- model_outputs------")
	print(lstm_model.outputs)
	print("---------------------")
	with open(model_history_path) as json_file:
		train_history = json.load(json_file)
	with open(config_file_path) as config_file:
		parameters = json.load(config_file)
	return lstm_model, train_history, parameters


def plot_results(results_path, history, mode="loss"):
	"""
    Plot the loss + metric training and validation results for the given model
    """
	if mode == "loss":
		plt.figure(figsize=(10, 10))
		plt.plot(history[mode][1:])
		plt.plot(history["val_" + mode][1:])
		plt.title("Model MSE " + mode)
		plt.xlabel("# Epochs")
		plt.ylabel(mode + " score")
		plt.legend(["train", "Validation"], loc="upper right", fontsize=14)
		plt.savefig(results_path)
		plt.show()
	elif mode == "metric":
		metric_name = "rmse"
		plt.figure(figsize=(10, 10))
		plt.plot(history[metric_name][1:])
		plt.plot(history["val_" + metric_name][1:])
		plt.title("Model " + metric_name + " metric")
		plt.xlabel("# Epochs")
		plt.ylabel(metric_name + " score")
		plt.legend(["train", "Validation"], loc="upper right", fontsize=14)
		plt.savefig(results_path)
		plt.show()
	else:
		print("A valid mode must be selected! Valid modes: 'loss' or 'metric'")


def evaluate_models(lstm_model, x_test, y_test, params, history, out_directory):
	"""
    Evaluates the previously trained modelsFull
    """
	loss_plot_name = "model_loss.png"
	results_loss_path = os.path.join(out_directory, loss_plot_name)
	plot_results(results_loss_path, history, mode="loss")
	metrics_plot_name = "model_metric.png"
	result_metric_path = os.path.join(out_directory, metrics_plot_name)
	plot_results(result_metric_path, history, mode="metric")
	test_loss, test_metric = lstm_model.evaluate(x=x_test, y=y_test, batch_size=params["test_batch_size"], verbose=1)
	return test_loss, test_metric


def predict_sequence(lstm_model, source_trajectory):
	"""
    Predicts the next n_steps_out bounding boxes using pre-trained models
    """
	src_trajectory_batch = np.expand_dims(source_trajectory, axis=0)
	predicted_trajectory = lstm_model.predict(src_trajectory_batch)
	return np.array(predicted_trajectory)


def path_predictor(lstm_model, params, x_test, y_test):
	"""
    Predicts target trajectories given a list of input trajectories
    """
	num_test_samples = x_test.shape[0]
	random_trajectory_samples = np.random.randint(low=0, high=num_test_samples, size=params["num_test_predictions"])
	sorted_random_trajectory_samples = sorted(random_trajectory_samples)
	source_trajectories = x_test[sorted_random_trajectory_samples]
	target_trajectories = y_test[sorted_random_trajectory_samples]
	predicted_trajectories = []
	for idx, source_trajectory in enumerate(source_trajectories):
		print("Source trajectory # {}: \n {}".format(idx, source_trajectory))
		print("Target trajectory # {}: \n {}".format(idx, target_trajectories[idx]))
		predictions = predict_sequence(lstm_model, source_trajectory=source_trajectory)
		predictions = predictions.astype("uint16")
		predicted_trajectories.append(predictions)
		print("Predicted trajectory # {}: \n {}".format(idx, predictions))


def main(args):
	"""
    Main function
    """
	in_hdf5_path = args.dataset
	saved_models_base_path = args.output
	config_file_path = args.config
	x_test, y_test = load_datasets(in_hdf5_path, partition='test')
	lstm_model_path = os.path.join(saved_models_base_path, "lstm_model.h5")
	model_history_path = os.path.join(saved_models_base_path, "model_history.json")
	lstm_model, train_history, params = load_models(lstm_model_path, model_history_path, config_file_path)
	test_loss, test_metric = evaluate_models(lstm_model, x_test, y_test, params, train_history, saved_models_base_path)
	print("MODEL EVALUATION RESULTS: \n ")
	print('--> Model Test loss value:', test_loss)
	print('--> Model Test metric value:', test_metric)
	path_predictor(lstm_model, params, x_test, y_test)


if __name__ == "__main__":
	""" This is executed when run from the command line """
	parser = argparse.ArgumentParser(description="CLI for making Path Predictions using trained models")
	parser.add_argument("dataset", help=" Path to input test dataset in .hdf5 format")
	parser.add_argument("output", help="Path to the output directory where resulting training graphs and prediction results are computed.")
	parser.add_argument("config", help="Path to configuration file used for testing.")
	args = parser.parse_args()
	main(args)
