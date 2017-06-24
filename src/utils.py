from datetime import datetime


def get_input_data_path(model, data_base_dir):
	if model == "big":
		# this is without time sampling
		sub_volumes_dir = data_base_dir + '/' + 'sub_volumes_except_time'
	elif model == "small":
		# this is without time sampling
		sub_volumes_dir = data_base_dir + '/' + 'sub_volumes_except_time'
	elif model == "freqSumSmall":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSumBig":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSum_TiedWeight":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSum_TiedWeight_Big":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	else:
		raise Exception("Invalid model name")

	return sub_volumes_dir



def get_data_path_with_timestamp(model, data_base_dir):
	time_postfix = datetime.now().strftime('%Y-%m-%d-%H%M%S')
	return data_base_dir + '/' + model + '/' + time_postfix
