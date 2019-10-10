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

from keras.backend.tensorflow_backend import set_session
from keras.utils import HDF5Matrix
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Dense, TimeDistributed, LeakyReLU, PReLU, Bidirectional, Dropout
from keras.optimizers import adam, nadam
from keras_radam import RAdam
from keras_lookahead import Lookahead
from keras.callbacks import EarlyStopping, ModelCheckpoint #,TensorBoard
from keras import backend

# Configure GPU memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)

sys.path.append('..')

def load_datasets(in_h5_path, partition ='train'):
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
        X_train = HDF5Matrix(datapath=in_h5_path, dataset = "train/X_train")
        y_train = HDF5Matrix(datapath=in_h5_path, dataset = "train/y_train")
        print(" Datasets successfully loaded!")
        print(" X_train shape : {} \n Y_train shape : {}".format(X_train.shape, y_train.shape))
        return X_train, y_train

    elif partition == 'test':
        X_test = HDF5Matrix(datapath=in_h5_path, dataset = "test/X_test")
        y_test = HDF5Matrix(datapath=in_h5_path, dataset = "test/y_test")
        print(" Datasets successfully loaded!")
        print(" X_test shape : {} \n Y_test shape : {}".format(X_test.shape, y_test.shape))
        return X_test, y_test
        
    else:
        print("Invalid 'partition' parameter: Valid values: ['train', 'test']")

def create_models(X_train, y_train, hyperparams):
    """
    Instantiate full_model, encoder_model and decoder_model
    
    Parameters
    ----------
    X_train -> HDF5_Group : HDF5_Group containing input train trajectories 
    y_train -> HDF5_Group : HDF5_Group containing target training trajectories
    hyperparams -> dict : Dictionary containing all hyperparamers for training/testing/predicting
        
    Returns
    ------
    lstm_model -> keras model : Keras LSTM model for trajectory prediction 
    """
    latent_dim_lstm_1 = hyperparams["latent_dim_lstm_1"]
    latent_dim_lstm_2 = hyperparams["latent_dim_lstm_2"]
    latent_dim_lstm_3 = hyperparams["latent_dim_lstm_3"]
    hidden_dense_dim_0 = hyperparams["hidden_dense_dim_0"]
    hidden_dense_dim_1 = hyperparams["hidden_dense_dim_1"]
    activation_dense_type = hyperparams["activation_dense_type"]
    leaky_alpha_rate = hyperparams["leaky_alpha_rate"]
    stateful_lstm_flag = hyperparams["stateful_lstm_flag"]
    bidirectional_lstm_flag = hyperparams["bidirectional_lstm_flag"]
    bidirectional_merge_mode = hyperparams["bidirectional_merge_mode"]
    dropout_flag = hyperparams["dropout_flag"]
    dropout_hidden_rate = hyperparams["dropout_hidden_rate"]
    dropout_output_rate = hyperparams["dropout_output_rate"]
    input_feature_vect_length = X_train.shape[2]  
    output_feature_vect_length = y_train.shape[2]
    # LSTM inputs
    lstm_inputs = Input(shape = (None, input_feature_vect_length), name = "lstm_inputs")
    # Bidirectional or feedforward CuDNNLSTM cells
    if bidirectional_lstm_flag == True:
        # Bidirectional CuDNNLSTM layers
        lstm_1 = Bidirectional(CuDNNLSTM(units = latent_dim_lstm_1, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_1"), merge_mode = bidirectional_merge_mode, name = "Bidirectional_lstm_1")
        lstm_2 = Bidirectional(CuDNNLSTM(units = latent_dim_lstm_2, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_2"), merge_mode = bidirectional_merge_mode, name = "Bidirectional_lstm_2")
        lstm_3 = Bidirectional(CuDNNLSTM(units = latent_dim_lstm_3, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_3"), merge_mode = bidirectional_merge_mode, name = "Bidirectional_lstm_3")
    else:
        # Feedforward CuDNNLSTM layers
        lstm_1 = CuDNNLSTM(units = latent_dim_lstm_1, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_1")
        lstm_2 = CuDNNLSTM(units = latent_dim_lstm_2, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_2")
        lstm_3 = CuDNNLSTM(units = latent_dim_lstm_3, return_sequences = True, stateful = stateful_lstm_flag, name = "lstm_3")

    # Time distributed Feed-Forward Network depending on activation function to choose
    if activation_dense_type == "selu":
        dense_hidden_0 = TimeDistributed(Dense(hidden_dense_dim_0, activation = activation_dense_type, kernel_initializer = 'lecun_normal', name = "dense_hidden_0"), name = "time_distributed_0")
        dense_hidden_1 = TimeDistributed(Dense(hidden_dense_dim_1, activation = activation_dense_type, kernel_initializer = 'lecun_normal', name = "dense_hidden_1"), name = "time_distributed_1")
        dense_output   = TimeDistributed(Dense(output_feature_vect_length, activation = "linear", name = "x_dense_output"), name = "time_distributed_output")
    elif activation_dense_type == "relu":
        dense_hidden_0 = TimeDistributed(Dense(hidden_dense_dim_0, activation = activation_dense_type, kernel_initializer = 'glorot_uniform', name = "dense_hidden_0"), name = "time_distributed_0")
        dense_hidden_1 = TimeDistributed(Dense(hidden_dense_dim_1, activation = activation_dense_type, kernel_initializer = 'glorot_uniform', name = "dense_hidden_1"), name = "time_distributed_1")
        dense_output   = TimeDistributed(Dense(output_feature_vect_length, activation = "linear", name = "x_dense_output"), name = "time_distributed_output")
    elif activation_dense_type == "LeakyReLU":
        dense_hidden_0 = TimeDistributed(Dense(hidden_dense_dim_0, activation = None, kernel_initializer = 'glorot_uniform', name = "dense_hidden_0"), name = "time_distributed_0")
        activation_0   = TimeDistributed(LeakyReLU(alpha=leaky_alpha_rate), name = "leaky_activation_0")
        dense_hidden_1 = TimeDistributed(Dense(hidden_dense_dim_1, activation = None, kernel_initializer = 'glorot_uniform', name = "dense_hidden_1"), name = "time_distributed_1")
        activation_1   = TimeDistributed(LeakyReLU(alpha=leaky_alpha_rate), name = "leaky_activation_1")
        dense_output   = TimeDistributed(Dense(output_feature_vect_length, activation = "linear", name = "x_dense_output"), name = "time_distributed_output")
    elif activation_dense_type == "PReLU":
        dense_hidden_0 = TimeDistributed(Dense(hidden_dense_dim_0, activation = None, kernel_initializer = 'glorot_uniform', name = "dense_hidden_0"), name = "time_distributed_0")
        activation_0   = TimeDistributed(PReLU(alpha_initializer='zeros', name = "prelu_activation_0"), name = "time_distributed_0_0")
        dense_hidden_1 = TimeDistributed(Dense(hidden_dense_dim_1, activation = None, kernel_initializer = 'glorot_uniform', name = "dense_hidden_1"), name = "time_distributed_1")
        activation_1   = TimeDistributed(PReLU(alpha_initializer='zeros', name = "prelu_activation_1"), name = "time_distributed_1_1")
        dense_output   = TimeDistributed(Dense(output_feature_vect_length, activation = "linear", name = "x_dense_output"), name = "time_distributed_output")
    else:
        'Please enter a valid activation function!  Avalaible : ["relu", "selu", "LeakyReLU", "PReLU"]'
    # If we want to apply dropout between densely connected layers
    if dropout_flag == True:
        dropout_1 = TimeDistributed(Dropout(dropout_hidden_rate, name = "dropout_1"), name = "dropout_hidden_time_distributed")
        dropout_2 = TimeDistributed(Dropout(dropout_output_rate, name = "dropout_2"), name = "dropout_output_time_distributed")
        if (activation_dense_type == 'PReLU' or activation_dense_type == "LeakyReLU"):
            lstm_1_outputs = lstm_1(lstm_inputs)
            lstm_2_outputs = lstm_2(lstm_1_outputs)
            lstm_3_outputs = lstm_3(lstm_2_outputs)
            dense_hidden_0 = dense_hidden_0(lstm_3_outputs)
            dense_hidden_0_outputs = activation_0(dense_hidden_0)
            dropout_hidden = dropout_1(dense_hidden_0_outputs)
            dense_hidden_1 = dense_hidden_1(dropout_hidden)
            dense_hidden_1_outputs = activation_1(dense_hidden_1)
            dropout_output = dropout_2(dense_hidden_1_outputs)
            dense_outputs = dense_output(dropout_output) 
        else:
            lstm_1_outputs = lstm_1(lstm_inputs)
            lstm_2_outputs = lstm_2(lstm_1_outputs)
            lstm_3_outputs = lstm_3(lstm_2_outputs)
            dense_hidden_0_outputs = dense_hidden_0(lstm_3_outputs)
            dropout_hidden = dropout_1(dense_hidden_0_outputs)
            dense_hidden_1_outputs = dense_hidden_1(dropout_hidden)
            dropout_output = dropout_2(dense_hidden_1_outputs)
            dense_outputs = dense_output(dropout_output)
    # Recurrent Model instantiation
    else:
        if (activation_dense_type == 'PReLU' or activation_dense_type == "LeakyReLU"):
            lstm_1_outputs = lstm_1(lstm_inputs)
            lstm_2_outputs = lstm_2(lstm_1_outputs)
            lstm_3_outputs = lstm_3(lstm_2_outputs)
            dense_hidden_0 = dense_hidden_0(lstm_3_outputs)
            dense_hidden_0_outputs = activation_0(dense_hidden_0)
            dense_hidden_1 = dense_hidden_1(dense_hidden_0_outputs)
            dense_hidden_1_outputs = activation_1(dense_hidden_1)
            dense_outputs = dense_output(dense_hidden_1_outputs) 
        else:
            lstm_1_outputs = lstm_1(lstm_inputs)
            lstm_2_outputs = lstm_2(lstm_1_outputs)
            lstm_3_outputs = lstm_3(lstm_2_outputs)
            dense_hidden_0_outputs = dense_hidden_0(lstm_3_outputs)
            dense_hidden_1_outputs = dense_hidden_1(dense_hidden_0_outputs)
            dense_outputs = dense_output(dense_hidden_1_outputs)

    # FULL LSTM MODEL 
    lstm_model = Model(inputs = lstm_inputs, outputs = dense_outputs)
    
    print(" Model successfully created!!!")
    return lstm_model

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

def compile_models(lstm_model, hyperparams):
    """
    Compile full training model with choosen hyperparameters.
    
    Parameters
    ----------
    lstm_model -> Keras model : Full training model to be compiled
    hyperparams -> dict : Dictionary containing all hyperparamers for model compilation
        
    Returns
    -------
    lstm_model (compiled) -> keras compiled model : A fully-compiled keras LSTM model using the specified hyperparameters.
    """
    clip_select_flag = hyperparams["clip_select_flag"]
    learning_rate = hyperparams["learning_rate"]
    BETA_1 = hyperparams["BETA_1"]
    BETA_2 = hyperparams["BETA_2"]
    lr_decay = hyperparams["lr_decay"]
    slow_weights = hyperparams["slow_weights_lookahead"]
    sync_lookahead = hyperparams["sync_lookahead"]
    warmup_RAdam = hyperparams["warmup_RAdam"]
    min_lr_RAdam = hyperparams["min_lr_RAdam"]
    weight_decay_RAdam = hyperparams["weight_decay_RAdam"]
    total_steps_RAdam = hyperparams["total_steps_RAdam"]
    clip_norm_thresh = hyperparams["clip_norm_thresh"]
    clip_val_thresh = hyperparams["clip_val_thresh"]
    ams_grad_flag = hyperparams["ams_grad_flag"]
    loss_function = hyperparams["loss_fcn"]
    optimizer = hyperparams["optimizer"]
    epsilon = hyperparams["epsilon"]
    
    if(clip_select_flag == "norm" and optimizer == "adam"):
        opt_norm_clip = Lookahead(keras.optimizers.adam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, decay = lr_decay, amsgrad = ams_grad_flag), sync_period = sync_lookahead, slow_step = slow_weights)
        lstm_model.compile(optimizer = opt_norm_clip, loss = loss_function, metrics = [rmse])
        return lstm_model
    
    if(clip_select_flag == "norm" and optimizer == "nadam"):
        opt_norm_clip = keras.optimizers.nadam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, epsilon = epsilon, schedule_decay = lr_decay)
        lstm_model.compile(optimizer = opt_norm_clip, loss = loss_function, metrics = [rmse])
        return lstm_model

    elif(clip_select_flag == "value" and optimizer == "adam"):
        opt_val_clip = Lookahead(keras.optimizers.adam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, decay = lr_decay, clipvalue = clip_val_thresh, amsgrad = ams_grad_flag), sync_period = sync_lookahead, slow_step = slow_weights)
        lstm_model.compile(optimizer = opt_val_clip, loss = loss_function, metrics = [rmse])
        return lstm_model

    elif(clip_select_flag == "value" and optimizer == "nadam"):
        opt_val_clip = keras.optimizers.nadam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, epsilon = epsilon, schedule_decay = lr_decay)
        lstm_model.compile(optimizer = opt_val_clip, loss = loss_function, metrics = [rmse])
        return lstm_model

    elif(optimizer == "RAdam"):
        opt = RAdam(learning_rate = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, epsilon = epsilon, weight_decay = weight_decay_RAdam, amsgrad = ams_grad_flag, total_steps = total_steps_RAdam, warmup_proportion = warmup_RAdam, min_lr = min_lr_RAdam)
        lstm_model.compile(optimizer = opt, loss = loss_function, metrics = [rmse]) 
        return lstm_model

    elif(optimizer == "Ranger"):
        opt = Lookahead(RAdam(learning_rate = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, epsilon = epsilon, weight_decay = weight_decay_RAdam, amsgrad = ams_grad_flag, total_steps = total_steps_RAdam, warmup_proportion = warmup_RAdam, min_lr = min_lr_RAdam), sync_period = sync_lookahead, slow_step = slow_weights)
        lstm_model.compile(optimizer = opt, loss = loss_function, metrics = [rmse]) 
        return lstm_model

    else:
        print(" Clipping Method OR Optimizer Selected is not avalaible! Please enter a valid string for these parameter: \n Valid Clipping:['norm', 'value'] \n Valid Optimizers: ['adam', 'NAdam', 'RAdam', 'Ranger']")
        return lstm_model

def fit_models(lstm_model, hyperparams, out_directory, X_train, y_train):
    """
    Fit training data with choosen hyperparameters.
    
    Parameters
    ----------
    lstm_model -> Keras model : A fully-compiled keras LSTM model using the specified hyperparameters
    out_directory -> str :Path to the output directory where resulting models, graphs and history results will be saved
    X_train -> HDF5_Group : HDF5_Group containing input training trajectories 
    y_train -> HDF5_Group : HDF5_Group containing target training trajectories
    hyperparams -> dict : Dictionary containing all hyperparamers for model fitting
        
    Returns
    ------
    training_history -> json : A .json file containing all training and validation information from the fitting process.
    """
    patience_steps = hyperparams["patience_steps"]
    batch_size = hyperparams["batch_size"]
    epochs = hyperparams["epochs"]
    val_split_size = hyperparams["val_split_size"]
    ## ------------ Keras Callbacks ------------------
    # Early Stopping
    early_stop = EarlyStopping(monitor = 'val_rmse', mode = 'min', patience = patience_steps, verbose = 1)
    # Model Checkpoint
    checkpoint_filename = "lstm_model.h5"
    checkpoint_path = os.path.join(out_directory, checkpoint_filename)
    model_ckpnt = ModelCheckpoint(filepath = checkpoint_path, monitor = 'val_rmse', mode = 'min', save_best_only = True, save_weights_only = False, verbose = 1)
    ## TensorBoard
    #tb_dir_name = "tb_logs"
    #tb_path = os.path.join(out_directory, tb_dir_name)
    #print( " Tensorboard's log directory : " + tb_path)
    #tensorboard = TensorBoard(log_dir = tb_path, histogram_freq = 1, batch_size = batch_size, write_graph = True, write_images = True, update_freq = "epoch")
    ## Fitting Function
    training_history = lstm_model.fit(x = X_train, y = y_train, shuffle = "batch", batch_size = batch_size, epochs = epochs, validation_split = val_split_size, callbacks = [early_stop, model_ckpnt])

    return training_history

def main(args):
    """
    Main function takes the parsed arguments to train Path Prediction LSTM. 
    
    Parameters
    ----------
    args.dataset : Path to input train-test dataset in .hdf5 format
    args.output : Path to the output directory where resulting models, graphs and history results are saved.
    args.config : Path to configuration file used for training
    
    Returns
    -------
    args.output folder containing saved best models(full_model, encoder_model and decoder_model) and model training history
    """
    IN_HDF5_PATH = args.dataset
    saved_models_base_path = args.output
    config_file_path = args.config
    X_train, y_train = load_datasets(IN_HDF5_PATH, partition = 'train')
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)
    lstm_model = create_models(X_train, y_train, hyperparams)
    lstm_model = compile_models(lstm_model, hyperparams)
    lstm_model.summary(line_length = 180)
    saved_model_dir_name = "ARCH-{}_Data-{}__bs-{}_lr-{}_loss-{}_opt-{}_BD-{}_BDmrg-{}_amsG-{}_DP-{}_sw-{}_sync-{}_act-{}_minLR-{}_ptc-{}_ep-{}".format(hyperparams["ARCH_ID"],
                                                                                                                               hyperparams["DATA_ID"],
                                                                                                                               hyperparams["batch_size"], 
                                                                                                                               hyperparams["learning_rate"], 
                                                                                                                               hyperparams["loss_fcn"], 
                                                                                                                               hyperparams["optimizer"],
                                                                                                                               hyperparams["bidirectional_lstm_flag"],
                                                                                                                               hyperparams["bidirectional_merge_mode"],
                                                                                                                               hyperparams["ams_grad_flag"],
                                                                                                                               hyperparams["dropout_flag"],
                                                                                                                               hyperparams["slow_weights_lookahead"],
                                                                                                                               hyperparams["sync_lookahead"],
                                                                                                                               hyperparams["activation_dense_type"],
                                                                                                                               hyperparams["min_lr_RAdam"],
                                                                                                                               hyperparams["patience_steps"], 
                                                                                                                               hyperparams["epochs"])
    output_path = os.path.join(saved_models_base_path, saved_model_dir_name)
    try:
        os.mkdir(output_path)
    except OSError as error: 
        print(error)
    print( "Overview hyperparameters used on training : ", saved_model_dir_name)
    history = fit_models(lstm_model, hyperparams, output_path, X_train, y_train)
    history_model_filename = "model_history.json"
    history_model_path = os.path.join(output_path, history_model_filename)
    with open(history_model_path, 'w') as f:
        json.dump(history.history, f)
    print(" Model Successfully Trained and Saved at {}".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ="CLI for Training Path Prediction LSTM models")
    parser.add_argument("dataset", help ="Path to input train-test dataset in .hdf5 format")
    parser.add_argument("output", help ="Path to the output directory where resulting models, graphs and history results are saved.")
    parser.add_argument("config", help ="Path to configuration file used for training.")
    args = parser.parse_args()
    main(args)