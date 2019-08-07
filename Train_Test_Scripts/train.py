#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marlon Andres Cajamarca Vega

Path Prediction LSTM - Path Prediction RNN Training script
"""
import keras
import os
import json
import sys
import argparse
import datetime

from keras.utils import HDF5Matrix
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Dense, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend

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

def create_models(encoder_input_train, decoder_input_train, hyperparams):
    """
    Instantiate full_model, encoder_model and decoder_model
    
    Parameters
    ----------
    encoder_input_train -> HDF5_Group : HDF5_Group containing train encoder_input dataset 
    decoder_input_train -> HDF5_Group : HDF5_Group containing train/test decoder_input dataset
    hyperparams -> dict : Dictionary containing all hyperparamers for training/testing/predicting
        latent_dim -> int : Number of Neurons for LSTM encoder and decoder cells.
        hidden_dense_dim -> int : Number of Neurons for hidden Dense layer
    Returns
    ------
    full_model -> keras model : Keras model for the complete encoder-decoder model training
    encoder_model -> keras model : Keras model for the inference encoder
    decoder_model -> keras_model : Keras model for the inference decoder
    """
    latent_dim = hyperparams["latent_dim"]
    hidden_dense_dim = hyperparams["hidden_dense_dim"]
    """ TRAINING ENCODER """
    input_feature_vect_length = encoder_input_train.shape[2]  
    output_feature_vect_lenght = decoder_input_train.shape[2]
    encoder_inputs = Input(shape = (None, input_feature_vect_length), name = "encoder_inputs")
    encoder_lstm = CuDNNLSTM(units = latent_dim, return_state = True, name = "encoder_lstm")
    # Making predictions with encoder model
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    """ TRAINING DECODER """
    decoder_inputs = Input(shape = (None, output_feature_vect_lenght), name = "decoder_inputs")
    decoder_lstm = CuDNNLSTM(units = latent_dim, return_sequences = True, return_state = True, name = "decoder_lstm")
    decoder_dense_hidden = TimeDistributed(Dense(latent_dim, activation = "linear", name = "decoder_dense_hidden_1"), name = "time_distributed_1")
    decoder_dense_hidden_2 = TimeDistributed(Dense(hidden_dense_dim, activation = "linear", name = "decoder_dense_hidden_2"), name = "time_distributed_2")
    decoder_dense_output = TimeDistributed(Dense(output_feature_vect_lenght, activation = "linear", name = "decoder_dense_output"), name = "time_distributed_output")
    # Making predictions with decoder model
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
    decoder_outputs_dense = decoder_dense_hidden(decoder_outputs)
    decoder_outputs_dense_2 = decoder_dense_hidden_2(decoder_outputs_dense)
    decoder_outputs = decoder_dense_output(decoder_outputs_dense_2)
    """  FULL TRAINING ENCODER-DECODER MODEL """
    full_model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)
    """  DEFINING INFERENCE ENCODER  """
    encoder_model = Model(inputs = encoder_inputs, outputs = encoder_states)
    """  DEFINING INFERENCE DECODER """
    decoder_state_input_h = Input(shape=(latent_dim,), name = "inference_encoder_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name = "inference_encoder_input_c")
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs_dense = decoder_dense_hidden(decoder_outputs)
    decoder_outputs_dense_2 = decoder_dense_hidden_2(decoder_outputs_dense)
    decoder_outputs = decoder_dense_output(decoder_outputs_dense_2)

    decoder_model = Model(inputs= [decoder_inputs] + decoder_states_inputs, outputs = [decoder_outputs] + decoder_states)
    print(" Train and Inference models successfully created!!!")
    return full_model, encoder_model, decoder_model

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

def compile_models(full_train_model, hyperparams):
    """
    Compile full training model with choosen hyperparameters.
    
    Parameters
    ----------
    full_train_model -> Keras model : Full training model to be compiled
    hyperparams -> dict : Dictionary containing all hyperparamers for training/testing/predicting
        loss_function -> str : Training Loss function. I.e. "mse", "acc", etc.
        clip_select_flag -> str : Optimizer's parameter flag to select between different gradient clipping methods at training-time
        learning_rate -> float : Optimizer's parameter between [0,1] for controlling the learning rate of the models 
        BETA_1 -> float : Optimizer's parameter between [0,1] for controlling the momentum's learning rate of the models
        BETA_2 -> float : Optimizer's parameter between [0,1] for controlling the momentum's learning rate of the models
        lr_decay -> float : 
        clip_norm_thresh -> float : Parameter to control the clipping norm threshole.
        clip_val_thresh -> float : Parameter to control the clipping value threshole.
        ams_grad_flag -> bool : wheter to apply the ams_grad variant for the Adam Optimizer. Default to True.
    Returns
    -------
    full_train_model -> keras model : A fully-compiled keras model using the specified hyperparameters.
    """
    clip_select_flag = hyperparams["clip_select_flag"]
    learning_rate = hyperparams["learning_rate"]
    BETA_1 = hyperparams["BETA_1"]
    BETA_2 = hyperparams["BETA_2"]
    lr_decay = hyperparams["lr_decay"]
    clip_norm_thresh = hyperparams["clip_norm_thresh"]
    clip_val_thresh = hyperparams["clip_val_thresh"]
    ams_grad_flag = hyperparams["ams_grad_flag"]
    loss_function = hyperparams["loss_fcn"]
    if clip_select_flag == "norm":
        opt_norm_clip = keras.optimizers.adam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, decay = lr_decay, clipnorm = clip_norm_thresh, amsgrad = ams_grad_flag)
        full_train_model.compile(optimizer = opt_norm_clip, loss = loss_function, metrics = [rmse])   
    elif clip_select_flag == "value":
        opt_val_clip = keras.optimizers.adam(lr = learning_rate, beta_1 = BETA_1, beta_2 = BETA_2, decay = lr_decay, clipvalue = clip_val_thresh, amsgrad = ams_grad_flag)
        full_train_model.compile(optimizer = opt_val_clip, loss = loss_function, metrics = [rmse])
    else:
        print(" Clipping Method Selected is not avalaible! Please enter a valid string for this parameter: Valid strings set:['norm', 'value']")
    return full_train_model

def fit_models(all_models, hyperparams, out_directory, encoder_input_train, decoder_input_train, decoder_target_train):
    """
    Fit training data with choosen hyperparameters.
    
    Parameters
    ----------
    all_models -> list() : List containing previously compiled full_model, encoder_model and decoder_model.
    out_directory -> str :Path to the output directory where resulting models, graphs and history results are saved.
    encoder_input_train -> HDF5_Group : HDF5_Group containing train encoder_input dataset 
    decoder_input_train -> HDF5_Group : HDF5_Group containing train decoder_input dataset
    decoder_target_train -> HDF5_Group : HDF5_Group containing train decoder_targets dataset
    hyperparams -> dict : Dictionary containing all hyperparamers for training/testing/predicting
        patience_steps -> int : Early-Stopping parameter. Number of epochs with no improvement after which training will be stopped
        batch_size -> int : Number of samples per gradient update.
        epochs -> int : Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        val_split_size -> float : Parameter used for making the train and test dataset splitting process. It is a value in the range [0,1].
        
    Returns
    ------
    training_history -> json : A .json file containing all information form the training process.
    Note: Best trained models (full_model, encoder_model, decoder_model) will be saved at out_directory path using
          model checkpointing and Early Stopping with `save_best = True`
    """
    full_train_model = all_models[0]
    encoder_model = all_models[1]
    decoder_model = all_models[2]
    patience_steps = hyperparams["patience_steps"]
    batch_size = hyperparams["batch_size"]
    epochs = hyperparams["epochs"]
    val_split_size = hyperparams["val_split_size"]
    latent_dim = hyperparams["latent_dim"]
    ## ------------ Keras Callbacks ------------------
    # Early Stopping
    early_stop = EarlyStopping(monitor = 'val_rmse', mode = 'min', patience = patience_steps, verbose = 1)
    # Model Checkpoint
    checkpoint_filename = "full_model.h5"
    checkpoint_path = os.path.join(out_directory, checkpoint_filename)
    model_ckpnt = ModelCheckpoint(filepath=checkpoint_path, monitor = 'val_rmse', mode = 'min', save_best_only = True, save_weights_only = False, verbose = 1)
    ## Fitting Function
    training_history = full_train_model.fit(x = [encoder_input_train, decoder_input_train], y = decoder_target_train, shuffle = "batch", batch_size = batch_size, epochs = epochs, validation_split = val_split_size, callbacks = [early_stop, model_ckpnt])
    ## Save inference models
    encoder_model_filename = "encoder_model.h5"
    decoder_model_filename = "decoder_model.h5"
    encoder_model_path = os.path.join(out_directory, encoder_model_filename)
    decoder_model_path = os.path.join(out_directory, decoder_model_filename)
    encoder_model.save(encoder_model_path)
    decoder_model.save(decoder_model_path)
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
    encoder_input_train, decoder_input_train, decoder_target_train = load_datasets(IN_HDF5_PATH, partition = 'train')
    with open(config_file_path) as config_file:
        hyperparams = json.load(config_file)
    full_model, encoder_model, decoder_model = create_models(encoder_input_train, decoder_input_train, hyperparams)
    full_train_model = compile_models(full_model, hyperparams)
    full_train_model.summary(line_length = 180)
    encoder_model.summary(line_length = 180)
    decoder_model.summary(line_length = 180)
    all_models = [full_train_model, encoder_model, decoder_model]
    history = fit_models(all_models, hyperparams, saved_models_base_path, encoder_input_train, decoder_input_train, decoder_target_train)
    history_model_filename = "model_history.json"
    history_model_path = os.path.join(saved_models_base_path, history_model_filename)
    with open(history_model_path, 'w') as f:
        json.dump(history.history, f)
    print(" Models Successfully Trained and Saved at {}".format(saved_models_base_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ="CLI for Training Path Prediction LSTM models")
    parser.add_argument("dataset", help ="Path to input train-test dataset in .hdf5 format")
    parser.add_argument("output", help ="Path to the output directory where resulting models, graphs and history results are saved.")
    parser.add_argument("config", help ="Path to configuration file used for training.")
    args = parser.parse_args()
    main(args)