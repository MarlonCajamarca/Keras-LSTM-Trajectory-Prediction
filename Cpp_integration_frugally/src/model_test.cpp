// Author      : Marlon Cajamarca
// Description : Frugally-Deep Implementation of Path Prediction LSTM Neural Network, Ansi-style

#include <fdeep/fdeep.hpp>
#include <vector>
#include <fstream>
#include <iostream>

int main()
{
	const auto encoder_model = fdeep::load_model("fdeep_encoder_model.json");
	std::cout << "Encoder Model Loaded!" << std::endl;
	const auto decoder_model = fdeep::load_model("fdeep_decoder_model.json");
	std::cout << "Decoder Model Loaded!" << std::endl;
	fdeep::shape5 in_traj_shape(1,1,1,10,4);
	const std::vector<float> src_traj  = {1728, 715, 191, 221,
									1717, 710, 202, 215,
									1706, 704, 206, 198,
									1695, 700, 217, 196,
									1687, 696, 228, 183,
									1680, 689, 240, 181,
									1668, 668, 240, 198,
									1661, 668, 243, 194,
									1650, 664, 251, 189,
									1635, 660, 266, 181};
	const fdeep::shared_float_vec shared_traj(fplus::make_shared_ref<fdeep::float_vec>(src_traj));
	const fdeep::tensor5 encoder_inputs(in_traj_shape, shared_traj);
	std::cout << "Trajectory #0!" << fdeep::show_tensor5(encoder_inputs) << std::endl;
	const auto encoder_states = encoder_model.predict({encoder_inputs});
	std::cout << "h_enc: "<< fdeep::show_tensor5(encoder_states.at(0)) << std::endl;
	std::cout << "c_enc: "<< fdeep::show_tensor5(encoder_states.at(1)) << std::endl;
	fdeep::shape5 bbox_shape(1,1,1,1,4);
	const std::vector<float> SOS_token  = {9999, 9999, 9999, 9999};
	const fdeep::shared_float_vec shared_SOS_token(fplus::make_shared_ref<fdeep::float_vec>(SOS_token));
	fdeep::tensor5 target_seq(bbox_shape, shared_SOS_token);
	auto decoder_outputs = decoder_model.predict({target_seq, encoder_states.at(0), encoder_states.at(1)});
	std::cout << "h_dec: "<< fdeep::show_tensor5(decoder_outputs.at(1)) << std::endl;
	std::cout << "c_dec: "<< fdeep::show_tensor5(decoder_outputs.at(2)) << std::endl;
	std::cout << "Predicted next bounding box!" << fdeep::show_tensor5(decoder_outputs.at(0)) << std::endl;
}
