# OBJECT TRAJECTORY FORECASTING USING LSTM-BASED RECURRENT NEURAL NETWORKS

Training, testing and inference with multi-input multi-output LSTM-based Recurrent Neural Networks for trajectory forecasting 

# Setup Your Anaconda Environment

In order to use the training, testing, inference scripts and utility tools, you need to create/clone an Anaconda dedicated environment containing predifined packages (i.e. tensorflow-gpu). If you do not already have Anaconda installed on your machine, please follow the installation steps avalaible at: 
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Once you have successfully installed Anaconda, clone the working environment from a previously created Anaconda environment by using the `environment.yaml` file. Then, please activate your recently cloned environment.

	# Create a new environment based on `environment.yaml` file
	conda env create -f environment.yaml
	# Activate your recently created environment
	conda activate <my_environment>

## Training, Testing and Inference Workflow
A common processing workflow involves using several utility tools for generating train and test sets in a .h5 file, then, train and testing is performed using those sets. Next, we will explain some of the details and requirements needed for the end-to-end pipeline. The processing pipeline could be separated in the following steps:

1. Extract raw object trajectories obtained from a previous Detection + Tracking processing pipeline.
2. Pack all trajectories in a convenient `raw_paths.h5` file.
3. Transform and group all raw paths from `raw_paths.h5` into an `train_test.h5` dataset.
4. Train and evaluate a Path Prediction LSTM-based neural model, called `model.h5`, using `LSTM_trainer.py` script and `train_test.h5` dataset.

Optionally:

5. Export `model.h5` model as a `model.pb` freezed model, use `keras_to_tensorflow.py` script.
6. Deploy either your `model.h5` or `model.pb` trained models into an existing video processing pipeline. Trajectory predictions could be used to replace/boost tracking algoritms by incorporating approximated information of future localizations for each object in the scene.

A sample of a multi-object detection + Tracking + Counting pipeline using the LSTM-based trajectory forecasting model trained using the previous workflow:

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/zB_2q-UUZ4s/0.jpg)](http://www.youtube.com/watch?v=zB_2q-UUZ4s)

## End-to-End Workflow Explained 

### 1. Extract raw object trajectories obtained from a previous Detection + Tracking processing pipeline.
In order to train the trajectory forcasting model, you need several .csv files containing raw trajectories obtained from a previous Detection + Tracking processing pipeline, a D+T pipeline for short. The D+T pipeline is in charge of supplying raw D+T trajectories packed in a .csv file with an specific output format. 

For the sake of completeness, you can download a set of these .csv raw D+T trajectories [HERE!](https://drive.google.com/file/d/1ylOH7bLw50VIdxqh6QRApGAfPnXte2C2/view?usp=sharing)

All of the previously described .csv files must be in a folder for further use by `dataset_creator` utility tool.

### 2. Pack all trajectories in a convenient `raw_paths.h5` file
Once you have all D+T raw trajectories on an user-defined `cache_folder`, use `dataset_creator` utility tool to pack all raw trajectories into a `raw_paths.h5` file located at `output_folder = full/path/to/raw_paths.h5`:

	python dataset-creator cache_folder output_folder
	
The resulting `raw_paths.h5` will contain several object trajectories for furter steps.

### 3. Transform and group all raw paths from `raw_paths.h5` into an `train_test.h5` dataset
All raw paths from `raw_paths.h5` will be used for `dataset_transformer` utility tool in order to generate a train-test dataset for further training and inference tasks. In addition, internal processing, filtering and polynomial trajectory smoothing can be configured by the user using the associated `config_transformer.json` file. The tool receives a `raw_dataset = full/path/to/raw_paths.h5` path, an it will generate a train-test dataset in .h5 format at location `out_dataset = full/path/to/train-test.h5` using the `config_transformer.json` file avalaible at location `config = full/path/to/config_transformer.json`:

	python dataset_transformer.py raw_dataset out_dataset config
	
The resulting `train-test.h5` dataset will contain 2 HDF5 groups within it, namely `train` and `test`, and will be used for training and inference scripts to train and validate trajectory forecasting models

Again, for the sake of completeness, you can download an already transformed dataset containing roughly 10 million trajectories from 12 different object classes. This dataset is ready for further train/evaluation tasks if you want to avoid steps 1, 2 and 3 of the workflow and go directly to step 4. The transformed dataset can be download [HERE!](https://drive.google.com/file/d/16OZnWe7Y0ie3gEHVEGR8GAb22cwJdMVJ/view?usp=sharing)

### 4. Train and evaluate a Path Prediction LSTM-based neural model
By using the `LSTM_trainer.py` script, the previously generated `train-test.h5` dataset, and an user configuration file `config_lstm.json` you can train your own custom LSTM-based Trajectory prediction model. 

In addition, in order to evaluate the performance of the trained models, use the `evaluate_lstm.py` script using also the previously generated `train-test.h5` dataset, and the same user configuration file `config_lstm.json` used for training. The `evaluate_lstm.py` script contains code use for load datasets, load trained models and making inference with them.

# Project Tools and Utilities

## Training
CLI for Training Path Prediction LSTM-based neural models.
### Usage: 
	LSTM_trainer.py [-h] dataset  output  config
	Positional arguments:
	  dataset     Path to input train-test dataset in .hdf5 format
	  output      Path to the output directory where resulting models, graphs and history results are saved.
	  config      Path to configuration file used for training.
	optional arguments:
	  -h, --help        show this help message and exit
	  --use_checkpoint  Load and use a previously trained LSTM model to restart training from that file. A new and restored model will be created, preserving original
	   input model.
	      
### Example:
	python LSTM_trainer.py  path/to/training_lstm_dataset.hdf5  path/to/output/directory  config_lstm.json  --use_checkpoint

## Inference
Command line tool for making Path Predictions using trained models
### Usage:
	evaluate_lstm.py [-h] dataset output configuration_file
	positional arguments:
	  dataset     Path to input test dataset in .hdf5 format
	  output      Path to the output directory where resulting training graphs and prediction results are computed.
	  config      Path to configuration file used for testing.
	optional arguments:
	  -h, --help  show this help message and exit
### Example:
	python evaluate_lstm.py  path/to/test_lstm_dataset.hdf5 path/to/trained/model/output/directory config_lstm.json

# Utility Tools:
## Dataset Creator
  Command line utility to create a training dataset in HDF5 format for the track recurrent neural network from CSV files created by the classifier + Kalman filter tracker
  . Classifier is a video processing pipeline running object detection + object tracking + post-processing to get raw training trajectories for further training. 
### Usage:
	dataset-creator [-h] [-a] input_directory output
	positional arguments:
    	input_directory  Path of a directory with CSV files to extract tracks.
    	output           Path of the output training dataset file with .hdf5 extension.
	optional arguments:
    	-h, --help       show this help message and exit
    	-a, --append     Flag to indicate that an existing HDF5 file can be used and new datasets should be appended to it.
### Example:
	python dataset-creator path/to/raw/classifier/detection/files/ path/to/output_raw_dataset.hdf5

## Train and Test Dataset Transformer
Command line tool for generating train and test sets for LSTM-based models from an input raw dataset created using **Raw Dataset Creator tool**.
### Usage:
	dataset_transformer.py [-h] raw_dataset out_dataset configuration_file
	positional arguments:
	  raw_dataset  Path to input raw dataset in .h5 format
	  out_dataset  Path to output train/test dataset in .h5 format
	  config       Path to configuration file used for dataset transformer tool.
	optional arguments:
	  -h, --help            show this help message and exit
	  --norm_target_data    Normalizes or smooths target trajectory data using nth-degree polynomial regression. nth-degree parameter specified in config.json file
### Example:
	python dataset_transformer.py path/to/raw_dataset.hdf5 path/to/output/train_test_dataset.hdf5 config_transformer.json --norm_target_data

## HDF5 Dataset Merger
Command line tool for merging two HDF5 datasets already generated by `dataset_transformer.py` .
### Usage:
	dataset_merger.py [-h] input_dataset_1 input_dataset_2 out_dataset
	positional arguments:
	  input_dataset_1  Full path to first input  dataset in .hdf5 format to merge
	  input_dataset_2  Full path to second input dataset in .hdf5 format to merge
	  out_dataset       Path to output train/test dataset in .h5 format
	optional arguments:
	  -h, --help   show this help message and exit
### Example:
	python dataset_merger.py path/to/input_dataset_1.hdf5 path/to/input_dataset_2.hdf5 path/to/output/merged_dataset.hdf5

## Keras HDF5 to Tensorflow PB model converter (3rd Party Tool)
The `keras_to_tensorflow.py` is a CLI that converts a trained keras model into a ready-for-inference TensorFlow PB model.

#### Summary
- In the default behaviour, this tool **freezes** the nodes (converts all TF variables to TF constants), and saves the inference graph and weights into a binary protobuf (.pb) file. During freezing, TensorFlow also applies node pruning which removes nodes with no contribution to the output tensor.

- This tool supports multiple output networks and enables the user to rename the output tensors via the `--output_nodes_prefix` flag.
 
- If the `--output_meta_ckpt` flag is set, the checkpoint and metagraph files for TensorFlow will also be exported
which can later be used in the `tf.train.Saver` class to continue training.   

#### How to use
Keras models can be saved as a single [`.hdf5` or `h5`] file, which stores both the architecture and weights, using the `model.save()` function.
 This model can be then converted to a TensorFlow model by calling this tool as follows:

    python keras_to_tensorflow.py 
        --input_model="path/to/keras/model.h5" 
        --output_model="path/to/save/model.pb"
        
Keras models can also be saved in two separate files where a [`.hdf5` or `h5`] file stores the weights, using the `model.save_weights()` function, and another `.json` file stores the network architecture using the `model.to_json()` function.
In this case, the model can be converted as follows:

    python keras_to_tensorflow.py 
        --input_model="path/to/keras/model.h5" 
        --input_model_json="path/to/keras/model.json" 
        --output_model="path/to/save/model.pb"

Try 

    python keras_to_tensorflow.py --help

to learn about other supported flags (quantize, output_nodes_prefix, save_graph_def).
#### Dependencies
--> Python version 3.X and above (packed into environment.yaml conda environment file)
- keras
- tensorflow
- absl
- pathlib
