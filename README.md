# LSTM-BASED PATH PREDICTION RECURRENT NEURAL NETWORKS

Training, testing and inference with LSTM-based Recurrent Neural Networks for path prediction forecasting. Aditionally, C++ integration prototipes are used to migrate Keras models into C++ aplication using Frugally-Deep tool.

# Setup Your Anaconda Environment

In order to use the training, testing and inference scripts, we need to create an Anaconda dedicated environment containing predifined packages (i.e. tensorflow-gpu). If you do not already have Anaconda installed on your machine, please follow the installation steps avalaible at: 
https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Once you have successfully installed Anaconda, create a new environment with from a previously created Anaconda environment by using the `environment.yaml` file. Then, please activate your recently created environment.

	# Create a new environment based on `environment.yaml` file
	conda env create -f environment.yaml
	# Activate your recently created environment
	conda activate <my_environment>

# Training, Testing and Inference 

## Training
CLI for Training Path Prediction LSTM-based neural models.
### Usage: 
	train.py [-h] dataset output config
	Positional arguments:
	  dataset     Path to input train-test dataset in .hdf5 format
	  output      Path to the output directory where resulting models, graphs and history results are saved.
	  config      Path to configuration file used for training.
	optional arguments:
	  -h, --help  show this help message and exit
### Example:
	python3 train.py /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Data/Data/train_test_datasets/v2_s4_10i_10o_FullLBV.hdf5 /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Data/Saved_Models/ config.json

## Inference
Command line tool for making Path Predictions using trained models
### Usage:
	predict.py [-h] dataset output config
	positional arguments:
	  dataset     Path to input test dataset in .hdf5 format
	  output      Path to the output directory where resulting training graphs and prediction results are computed.
	  config      Path to configuration file used for testing.
	optional arguments:
	  -h, --help  show this help message and exit
### Example:
	python3 predict.py /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Data/train_test_datasets/v2_s4_10i_10o_FullLBV.hdf5 /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Saved_Models/ config.json

# Utility Tools:

## Train and Test Dataset transformator
Command line tool for generating train and test sets for LSTM-based models from an input raw dataset created using **Raw Dataset Creator tool**.
### Usage:
	train_test_dataset_transformer.py [-h] raw_dataset out_dataset config
	positional arguments:
	  raw_dataset  Path to input raw dataset in .h5 format
	  out_dataset  Path to output train/test dataset in .h5 format
	  config       Path to configuration file used for dataset transformer tool.
	optional arguments:
	  -h, --help   show this help message and exit
### Example:
	python3 train_test_dataset_transformer.py /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Data/raw_datasets/v2_dataset.hdf5 /home/dev/git/aforos/utils/prototipos/LSTM_Prediccion_Trayectorias/Data/train_test_datasets/10i_10o_10s_full_dataset.hdf5 config_transformer.json

## Dataset Creator
  Command line utility to create a training dataset in HDF5 format for the track recurrent neural network from CSV files created by the classifier.
### Usage:
	dataset-creator [-h] [-a] input_directory output
	positional arguments:
    	input_directory  Path of a directory with CSV files to extract tracks.
    	output           Path of the output training dataset file with .hdf5 extension.
	optional arguments:
    	-h, --help       show this help message and exit
    	-a, --append     Flag to indicate that an existing HDF5 file can be used and new datasets should be appended to it.
### Example:
	python3 dataset-creator /home/dev/git/aforos/utils/Prototipos/Prototipo_Red_Recurrente_Prediccion_Trayectorias/Data/Raw_Yolo_Results/ /home/dev/git/aforos/utils/Prototipos/Prototipo_Red_Recurrente_Prediccion_Trayectorias/Data/raw_datasets/output.hdf5

# Installing and Running C++ Integration Scripts Using Frugally-Deep

Frugally-Deep is a Header-only library for using Keras models in C++. For a detailed information please refer yourself to: https://github.com/Dobiasd/frugally-deep
Fortunatelly, all layers used at model instantiation are supported at this time. However, some key functionalities are still to be added.

## Requirements and Installation
A C++14-compatible compiler is needed. Compilers from these versions on are fine: GCC 4.9, Clang 3.7 (libc++ 3.7) and Visual C++ 2015. In addition, You also can install frugally-deep using cmake. Detailed installation instructions are provided at https://github.com/Dobiasd/frugally-deep/blob/master/INSTALL.md. After this, all includes needed to run frugally-deep are avalaible under `/includes/usr/local/include/`. 

## Usage:
Use Keras/Python to build (`model.compile(...)`), train (`model.fit(...)`) and test (`model.evaluate(...)`) your model as usual. Then save it to a single HDF5 file using `model.save('....h5', include_optimizer=False)`. The `image_data_format` in your model must be `channels_last`, which is the default when using the TensorFlow backend. Models created with a different `image_data_format` and other backends are not supported.

In order to load the previously trained LSTM Encoder and Decoder models, Frugally-Deep library allows us to convert .h5 Keras trained models into .json format using its `convert.py` script. Convert models to the frugally-deep file format with `keras_export/convert_model.py`

Create a new C++ aplication project folder with a `/src` subfolder where your source files are avalaible (i.e. `/src/main.cpp`). Aditionally, place the previously converted models at the root of the project. An example od the folder hierarchy described could be:

	Project Folder/
	---> includes/
		---> /usr/local/include/
			---> Eigen/
			---> eigen3/
			---> fdeep/
			---> fplus/
			---> nlohmann/
	---> src/
		---> main.cpp
		---> another_source_file.cpp
	---> encoder_model.json
	---> decoder_model.json
	---> another_model.json 

Finally load the converted models by using `fdeep::load_model("encoder_model.json")` and then use `model.predict(...)` to invoke a forward-pass/inference-step with your data. For example:

	#include <fdeep/fdeep.hpp>
	#include <vector>
	#include <fstream>
	#include <iostream>
	int main()
	{
		// Loading the converted trained models
		const auto encoder_model = fdeep::load_model("fdeep_encoder_model.json");
		// Make inference with those models
		const auto encoder_states = encoder_model.predict({encoder_inputs});
		// Inspecting the results of the model prediction
		std::cout << "h_enc: "<< fdeep::show_tensor5(encoder_states.at(0)) << std::endl;
		std::cout << "c_enc: "<< fdeep::show_tensor5(encoder_states.at(1)) << std::endl;

Frequently asked questions and further implementation information and tips about Frugally-deep can be accessed at: https://github.com/Dobiasd/frugally-deep/blob/master/FAQ.md 