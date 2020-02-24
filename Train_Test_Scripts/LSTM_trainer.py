#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marlon Andres Cajamarca Vega

Path Prediction LSTM - Path Prediction RNN Training + Tuning Script
"""
import keras
import os
import json
import sys
import argparse
import tensorflow as tf
import time
from keras.backend.tensorflow_backend import set_session
from keras.utils import HDF5Matrix
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Dense, TimeDistributed, LeakyReLU, PReLU, Bidirectional, Dropout
from keras_radam import RAdam
from keras_lookahead import Lookahead
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from keras import backend

# Configure GPU memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)

sys.path.append('..')

_USE_TENSORBOARD = False
_DEFAULT_DATASET_PARTITION = "train"
_MODEL_FILENAME = "lstm_model.h5"
_MODEL_HISTORY_FILENAME = "model_history.json"
_MODEL_CSV_LOG_FILENAME = "training_history_logs.csv"


class LstmTrainer(object):

	def __init__(self, in_hdf5_path: str, saved_models_base_path: str, hyperparameters, use_checkpoint: bool):
		self.in_hdf5_path = in_hdf5_path
		self.saved_models_base_path = saved_models_base_path
		self.hyperparameters = hyperparameters
		self.partition = _DEFAULT_DATASET_PARTITION
		self.use_checkpoint = use_checkpoint
		self.x_train = None
		self.y_train = None
		self.x_test = None
		self.y_test = None
		self.lstm_model = None
		self.lstm_inputs = None
		self.lstm_1 = None
		self.lstm_2 = None
		self.lstm_3 = None
		self.lstm_1_outputs = None
		self.lstm_2_outputs = None
		self.lstm_3_outputs = None
		self.dense_hidden_0 = None
		self.activation_0 = None
		self.dense_hidden_1 = None
		self.activation_1 = None
		self.dense_output = None
		self.dropout_1 = None
		self.dropout_2 = None
		self.optimizer = None
		self.training_history = None
		self.history_model_path = str
		self.output_path = str

	def run(self):
		start_train_time = time.time()
		self.load_datasets()
		if self.use_checkpoint:
			self.load_models()
		else:
			self.create_models()
		self.compile_models()
		self.fit_models()
		end_train_time = time.time()
		print(" Model successfully trained in {} minutes!!!".format((end_train_time - start_train_time) / 60))

	def load_datasets(self):
		"""
		Load train or test dataset
		"""
		if self.partition == 'train':
			self.x_train = HDF5Matrix(datapath=self.in_hdf5_path, dataset="train/X_train")
			self.y_train = HDF5Matrix(datapath=self.in_hdf5_path, dataset="train/y_train")
			print(" Datasets successfully loaded!")
			print(" X_train shape : {} \n Y_train shape : {}".format(self.x_train.shape, self.y_train.shape))
		elif self.partition == 'test':
			self.x_test = HDF5Matrix(datapath=self.in_hdf5_path, dataset="test/X_test")
			self.y_test = HDF5Matrix(datapath=self.in_hdf5_path, dataset="test/y_test")
			print(" Datasets successfully loaded!")
			print(" X_test shape : {} \n Y_test shape : {}".format(self.x_test.shape, self.y_test.shape))
		else:
			print("Invalid 'partition' parameter: Valid values: ['train', 'test']")
			sys.exit(1)

	def load_models(self):
		output_model_name = self.get_model_name_from_configuration()
		output_path = os.path.join(self.saved_models_base_path, output_model_name)
		full_output_pathname = os.path.join(output_path, _MODEL_FILENAME)
		self.lstm_model = keras.models.load_model(full_output_pathname, custom_objects={'rmse': rmse, 'Lookahead': Lookahead, 'RAdam': RAdam})
		print(" --> LSTM model successfully restored from checkpoint file...")
		self.lstm_model.summary(line_length=180)

	def create_models(self):
		"""
		Instantiate full_model, encoder_model and decoder_model
		"""
		print(" --> LSTM model instantiation started!")
		# input layer
		self.lstm_inputs = Input(shape=(None, self.x_train.shape[2]), name="lstm_inputs")
		# LSTM layers
		self.lstm_layers_instantiation()
		# Dense layers
		if self.hyperparameters["activation_dense_type"] == "selu" or self.hyperparameters["activation_dense_type"] == "relu":
			self.activated_dense_layers_instantiation()
		else:
			# Leakyrelu or prelu
			self.non_activated_dense_layers_instantiation()
		# Instantiate model architecture
		if self.hyperparameters["dropout_flag"]:
			self.dropout_layers_instantiation()
			self.instantiate_model_with_dropout()
		else:
			self.instantiate_model_without_dropout()
		print(" --> LSTM model successfully created...")
		self.lstm_model.summary(line_length=180)

	def compile_models(self):
		"""
		Compile full training model with chosen hyperparameters.
		"""
		if self.hyperparameters["clip_select_flag"] == "norm" and self.hyperparameters["optimizer"] == "adam":
			self.optimizer = Lookahead(keras.optimizers.adam(lr=self.hyperparameters["learning_rate"],
			                                                 beta_1=self.hyperparameters["BETA_1"],
			                                                 beta_2=self.hyperparameters["BETA_2"],
			                                                 decay=self.hyperparameters["lr_decay"],
			                                                 amsgrad=self.hyperparameters["ams_grad_flag"]),
			                           sync_period=self.hyperparameters["sync_lookahead"],
			                           slow_step=self.hyperparameters["slow_weights_lookahead"])
		elif self.hyperparameters["clip_select_flag"] == "value" and self.hyperparameters["optimizer"] == "adam":
			self.optimizer = Lookahead(keras.optimizers.adam(lr=self.hyperparameters["learning_rate"],
			                                                 beta_1=self.hyperparameters["BETA_1"],
			                                                 beta_2=self.hyperparameters["BETA_2"],
			                                                 decay=self.hyperparameters["lr_decay"],
			                                                 clipvalue=self.hyperparameters["clip_val_thresh"],
			                                                 amsgrad=self.hyperparameters["ams_grad_flag"]),
			                           sync_period=self.hyperparameters["sync_lookahead"],
			                           slow_step=self.hyperparameters["slow_weights_lookahead"])
		elif (self.hyperparameters["clip_select_flag"] == "norm" or self.hyperparameters[
			"clip_select_flag"] == "value") and self.hyperparameters["optimizer"] == "nadam":
			self.optimizer = keras.optimizers.nadam(lr=self.hyperparameters["learning_rate"],
			                                        beta_1=self.hyperparameters["BETA_1"],
			                                        beta_2=self.hyperparameters["BETA_2"],
			                                        epsilon=self.hyperparameters["epsilon"],
			                                        schedule_decay=self.hyperparameters["lr_decay"])
		elif self.hyperparameters["optimizer"] == "RAdam":
			self.optimizer = RAdam(learning_rate=self.hyperparameters["learning_rate"],
			                       beta_1=self.hyperparameters["BETA_1"],
			                       beta_2=self.hyperparameters["BETA_2"],
			                       epsilon=self.hyperparameters["epsilon"],
			                       weight_decay=self.hyperparameters["weight_decay_RAdam"],
			                       amsgrad=self.hyperparameters["ams_grad_flag"],
			                       total_steps=self.hyperparameters["total_steps_RAdam"],
			                       warmup_proportion=self.hyperparameters["warmup_RAdam"],
			                       min_lr=self.hyperparameters["min_lr_RAdam"])
		elif self.hyperparameters["optimizer"] == "Ranger":
			self.optimizer = Lookahead(RAdam(learning_rate=self.hyperparameters["learning_rate"],
			                                 beta_1=self.hyperparameters["BETA_1"],
			                                 beta_2=self.hyperparameters["BETA_2"],
			                                 epsilon=self.hyperparameters["epsilon"],
			                                 weight_decay=self.hyperparameters["weight_decay_RAdam"],
			                                 amsgrad=self.hyperparameters["ams_grad_flag"],
			                                 total_steps=self.hyperparameters["total_steps_RAdam"],
			                                 warmup_proportion=self.hyperparameters["warmup_RAdam"],
			                                 min_lr=self.hyperparameters["min_lr_RAdam"]),
			                           sync_period=self.hyperparameters["sync_lookahead"],
			                           slow_step=self.hyperparameters["slow_weights_lookahead"])
		else:
			print(" Clipping Method OR Optimizer Selected is not available! ")
			print(
				" Please enter a valid string for these parameter: \n Valid Clipping:['norm', 'value'] \n Valid Optimizers: ['adam', 'NAdam', 'RAdam', 'Ranger']")
			sys.exit(1)
		self.lstm_model.compile(optimizer=self.optimizer, loss=self.hyperparameters["loss_fcn"], metrics=[rmse])

	def fit_models(self):
		"""
			Fit training data with chosen hyperparameters.
		"""
		model_filename = str
		output_model_name = self.get_model_name_from_configuration()
		self.output_path = os.path.join(self.saved_models_base_path, output_model_name)
		if not self.use_checkpoint:
			try:
				model_filename = _MODEL_FILENAME
				os.mkdir(self.output_path)
			except OSError as error:
				print(error)
		else:
			model_filename = "ckp_" + _MODEL_FILENAME
		full_out_pathname = os.path.join(self.output_path, model_filename)
		model_checkpoint = ModelCheckpoint(filepath=full_out_pathname, monitor='val_rmse', mode='min',
		                                   save_best_only=True, save_weights_only=False, verbose=1)
		early_stop = EarlyStopping(monitor='val_rmse', mode='min', patience=self.hyperparameters["patience_steps"],
		                           verbose=1)
		csv_logs_file_path = os.path.join(self.output_path, _MODEL_CSV_LOG_FILENAME)
		csv_logger = CSVLogger(csv_logs_file_path, append=True)
		if _USE_TENSORBOARD:
			print(" Model fitting using Tensorboard!")
			tensorboard_logs_filename = "tensorboard_logs"
			tensorboard_logs_path = os.path.join(self.output_path, tensorboard_logs_filename)
			print(" Tensorboard's log directory : " + tensorboard_logs_path)
			tensor_board = TensorBoard(log_dir=tensorboard_logs_path,
			                           histogram_freq=1,
			                           batch_size=self.hyperparameters["batch_size"],
			                           write_graph=True,
			                           write_images=True,
			                           update_freq="epoch")
			self.training_history = self.lstm_model.fit(x=self.x_train,
			                                            y=self.y_train,
			                                            shuffle="batch",
			                                            batch_size=self.hyperparameters["batch_size"],
			                                            epochs=self.hyperparameters["epochs"],
			                                            validation_split=self.hyperparameters["val_split_size"],
			                                            callbacks=[early_stop, model_checkpoint, csv_logger, tensor_board])
		else:
			self.training_history = self.lstm_model.fit(x=self.x_train,
			                                            y=self.y_train,
			                                            shuffle="batch",
			                                            batch_size=self.hyperparameters["batch_size"],
			                                            epochs=self.hyperparameters["epochs"],
			                                            validation_split=self.hyperparameters["val_split_size"],
			                                            callbacks=[early_stop, model_checkpoint, csv_logger])
		history_model_path = os.path.join(self.output_path, _MODEL_HISTORY_FILENAME)
		with open(history_model_path, 'w') as f:
			json.dump(self.training_history.history, f)
		print(" Model training history saved successfully!")
		print(" Model Successfully saved at {}".format(self.output_path))

	def lstm_layers_instantiation(self):
		if self.hyperparameters["bidirectional_lstm_flag"]:
			self.lstm_1 = Bidirectional(CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_1"],
			                                      return_sequences=True,
			                                      stateful=self.hyperparameters["stateful_lstm_flag"],
			                                      name="lstm_1"),
			                            merge_mode=self.hyperparameters["bidirectional_merge_mode"],
			                            name="Bidirectional_lstm_1")
			self.lstm_2 = Bidirectional(CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_2"],
			                                      return_sequences=True,
			                                      stateful=self.hyperparameters["stateful_lstm_flag"],
			                                      name="lstm_2"),
			                            merge_mode=self.hyperparameters["bidirectional_merge_mode"],
			                            name="Bidirectional_lstm_2")
			self.lstm_3 = Bidirectional(CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_3"],
			                                      return_sequences=True,
			                                      stateful=self.hyperparameters["stateful_lstm_flag"],
			                                      name="lstm_3"),
			                            merge_mode=self.hyperparameters["bidirectional_merge_mode"],
			                            name="Bidirectional_lstm_3")
		else:
			self.lstm_1 = CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_1"],
			                        return_sequences=True,
			                        stateful=self.hyperparameters["stateful_lstm_flag"],
			                        name="lstm_1")
			self.lstm_2 = CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_2"],
			                        return_sequences=True,
			                        stateful=self.hyperparameters["stateful_lstm_flag"],
			                        name="lstm_2")
			self.lstm_3 = CuDNNLSTM(units=self.hyperparameters["latent_dim_lstm_3"],
			                        return_sequences=True,
			                        stateful=self.hyperparameters["stateful_lstm_flag"],
			                        name="lstm_3")

	def activated_dense_layers_instantiation(self):
		if self.hyperparameters["activation_dense_type"] == "selu":
			self.dense_hidden_0 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_0"],
			                                            activation=self.hyperparameters["activation_dense_type"],
			                                            kernel_initializer='lecun_normal',
			                                            name="dense_hidden_0"),
			                                      name="time_distributed_0")
			self.dense_hidden_1 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_1"],
			                                            activation=self.hyperparameters["activation_dense_type"],
			                                            kernel_initializer='lecun_normal',
			                                            name="dense_hidden_1"),
			                                      name="time_distributed_1")
			self.dense_output = TimeDistributed(Dense(self.y_train.shape[2],
			                                          activation="linear",
			                                          name="x_dense_output"),
			                                    name="time_distributed_output")
		elif self.hyperparameters["activation_dense_type"] == "relu":
			self.dense_hidden_0 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_0"],
			                                            activation=self.hyperparameters["activation_dense_type"],
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_0"),
			                                      name="time_distributed_0")
			self.dense_hidden_1 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_1"],
			                                            activation=self.hyperparameters["activation_dense_type"],
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_1"),
			                                      name="time_distributed_1")
			self.dense_output = TimeDistributed(Dense(self.y_train.shape[2],
			                                          activation="linear",
			                                          name="x_dense_output"),
			                                    name="time_distributed_output")
		else:
			'Please enter a valid activation function!  Available : ["relu", "selu"]'
			sys.exit(1)

	def non_activated_dense_layers_instantiation(self):
		if self.hyperparameters["activation_dense_type"] == "LeakyReLU":
			self.dense_hidden_0 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_0"],
			                                            activation=None,
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_0"),
			                                      name="time_distributed_0")
			self.activation_0 = TimeDistributed(LeakyReLU(alpha=self.hyperparameters["leaky_alpha_rate"]),
			                                    name="leaky_activation_0")
			self.dense_hidden_1 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_1"],
			                                            activation=None,
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_1"),
			                                      name="time_distributed_1")
			self.activation_1 = TimeDistributed(LeakyReLU(alpha=self.hyperparameters["leaky_alpha_rate"]),
			                                    name="leaky_activation_1")
			self.dense_output = TimeDistributed(Dense(self.y_train.shape[2],
			                                          activation="linear",
			                                          name="x_dense_output"),
			                                    name="time_distributed_output")
		elif self.hyperparameters["activation_dense_type"] == "PReLU":
			self.dense_hidden_0 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_0"],
			                                            activation=None,
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_0"),
			                                      name="time_distributed_0")
			self.activation_0 = TimeDistributed(PReLU(alpha_initializer='zeros',
			                                          name="prelu_activation_0"),
			                                    name="time_distributed_0_0")
			self.dense_hidden_1 = TimeDistributed(Dense(self.hyperparameters["hidden_dense_dim_1"],
			                                            activation=None,
			                                            kernel_initializer='glorot_uniform',
			                                            name="dense_hidden_1"),
			                                      name="time_distributed_1")
			self.activation_1 = TimeDistributed(PReLU(alpha_initializer='zeros',
			                                          name="prelu_activation_1"),
			                                    name="time_distributed_1_1")
			self.dense_output = TimeDistributed(Dense(self.y_train.shape[2],
			                                          activation="linear",
			                                          name="x_dense_output"),
			                                    name="time_distributed_output")
		else:
			'Please enter a valid activation function!  Available:["LeakyReLU", "PReLU"]'
			sys.exit(1)

	def dropout_layers_instantiation(self):
		self.dropout_1 = TimeDistributed(Dropout(self.hyperparameters["dropout_hidden_rate"],
		                                         name="dropout_1"),
		                                 name="dropout_hidden_time_distributed")
		self.dropout_2 = TimeDistributed(Dropout(self.hyperparameters["dropout_output_rate"],
		                                         name="dropout_2"),
		                                 name="dropout_output_time_distributed")

	def instantiate_model_with_dropout(self):
		if self.hyperparameters["activation_dense_type"] == 'PReLU' or self.hyperparameters[
			"activation_dense_type"] == "LeakyReLU":
			lstm_1_outputs = self.lstm_1(self.lstm_inputs)
			lstm_2_outputs = self.lstm_2(lstm_1_outputs)
			lstm_3_outputs = self.lstm_3(lstm_2_outputs)
			dense_h0 = self.dense_hidden_0(lstm_3_outputs)
			dense_h0_outputs = self.activation_0(dense_h0)
			dropout_hidden = self.dropout_1(dense_h0_outputs)
			dense_h1 = self.dense_hidden_1(dropout_hidden)
			dense_h1_outputs = self.activation_1(dense_h1)
			dropout_output = self.dropout_2(dense_h1_outputs)
			dense_outputs = self.dense_output(dropout_output)
			self.lstm_model = Model(inputs=self.lstm_inputs, outputs=dense_outputs)
		else:
			# SELU and RELU activations layers
			lstm_1_outputs = self.lstm_1(self.lstm_inputs)
			lstm_2_outputs = self.lstm_2(lstm_1_outputs)
			lstm_3_outputs = self.lstm_3(lstm_2_outputs)
			dense_hidden_0_outputs = self.dense_hidden_0(lstm_3_outputs)
			dropout_hidden = self.dropout_1(dense_hidden_0_outputs)
			dense_hidden_1_outputs = self.dense_hidden_1(dropout_hidden)
			dropout_output = self.dropout_2(dense_hidden_1_outputs)
			dense_outputs = self.dense_output(dropout_output)
			self.lstm_model = Model(inputs=self.lstm_inputs, outputs=dense_outputs)

	def instantiate_model_without_dropout(self):
		if self.hyperparameters["activation_dense_type"] == 'PReLU' or self.hyperparameters["activation_dense_type"] == "LeakyReLU":
			lstm_1_outputs = self.lstm_1(self.lstm_inputs)
			lstm_2_outputs = self.lstm_2(lstm_1_outputs)
			lstm_3_outputs = self.lstm_3(lstm_2_outputs)
			dense_h0 = self.dense_hidden_0(lstm_3_outputs)
			dense_h0_outputs = self.activation_0(dense_h0)
			dense_h1 = self.dense_hidden_1(dense_h0_outputs)
			dense_h1_outputs = self.activation_1(dense_h1)
			dense_outputs = self.dense_output(dense_h1_outputs)
			self.lstm_model = Model(inputs=self.lstm_inputs, outputs=dense_outputs)
		else:
			# SELU and RELU activations layers
			lstm_1_outputs = self.lstm_1(self.lstm_inputs)
			lstm_2_outputs = self.lstm_2(lstm_1_outputs)
			lstm_3_outputs = self.lstm_3(lstm_2_outputs)
			dense_hidden_0_outputs = self.dense_hidden_0(lstm_3_outputs)
			dense_hidden_1_outputs = self.dense_hidden_1(dense_hidden_0_outputs)
			dense_outputs = self.dense_output(dense_hidden_1_outputs)
			self.lstm_model = Model(inputs=self.lstm_inputs, outputs=dense_outputs)

	def get_model_name_from_configuration(self):
		output_model_name = "ARCH-{}_Data-{}__bs-{}_lr-{}_loss-{}_opt-{}_BD-{}_BDmrg-{}_amsG-{}_DP-{}_sw-{}_sync-{}_act-{}_minLR-{}_ptc-{}_ep-{}".format(
			self.hyperparameters["ARCH_ID"],
			self.hyperparameters["DATA_ID"],
			self.hyperparameters["batch_size"],
			self.hyperparameters["learning_rate"],
			self.hyperparameters["loss_fcn"],
			self.hyperparameters["optimizer"],
			self.hyperparameters["bidirectional_lstm_flag"],
			self.hyperparameters["bidirectional_merge_mode"],
			self.hyperparameters["ams_grad_flag"],
			self.hyperparameters["dropout_flag"],
			self.hyperparameters["slow_weights_lookahead"],
			self.hyperparameters["sync_lookahead"],
			self.hyperparameters["activation_dense_type"],
			self.hyperparameters["min_lr_RAdam"],
			self.hyperparameters["patience_steps"],
			self.hyperparameters["epochs"])
		print("Overview hyperparameters used on training : ", output_model_name)
		return output_model_name


def rmse(y_true, y_prediction):
	"""
	Calculate Root Mean Squared Error custom metric
	"""
	return backend.sqrt(backend.mean(backend.square(y_prediction - y_true), axis=-1))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="CLI for Training Path Prediction LSTM models")
	parser.add_argument("dataset", help="Path to input train-test dataset in .hdf5 format")
	parser.add_argument("output",
	                    help="Path to the output directory where resulting models, graphs and history results are saved.")
	parser.add_argument("config", help="Path to configuration file used for training.")
	parser.add_argument("--use_checkpoint", action="store_true", default=False)
	args = parser.parse_args()
	with open(args.config) as config_file:
		hyperparameters = json.load(config_file)
	LstmTrainer(args.dataset, args.output, hyperparameters, args.use_checkpoint).run()
