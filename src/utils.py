from datetime import datetime
from models.fc_freqSum_Models import build_fc_freqSum_TiedWeight
from models.fc_freqSum_Models import build_fc_freqSum_TiedWeight_NoBias
from models.fc_freqSum_Models import build_fc_freqSum_TiedWeight_NoDropout
from models.fc_freqSum_Models import build_fc_freqSum
from models.fc_freqSum_Models import build_fc_freqSum_TiedWeight_Big
from models.fc_freqSum_Models import build_fc_freqSum_NoTiedWeight_Big
from models.fc_freqSum_Models import build_fc_freqSum_NoTiedWeight_BN_Small
from models.fc_freqSum_Models import build_fc_freqSum_NoTiedWeight_Small
#from models.fc_freqSum_Models import build_fc_freqSum_NoTiedWeight_Medium
from models.fc_freqSum_Models import build_fc_freqSum_NoTiedWeight_Tiny
from models.fc_freq_Models import build_fc_freq_4_30_NoTiedWeight_Small
from models.fc_freq_Models import build_fc_freq_4_30_TiedWeight_Small
from models.fc_freq_Models import build_fc_freq_5_TiedWeight_Small
from models.fc_freq_Models import build_fc_freq_5_NoTiedWeight_Small
from models.fc_freq_Models import build_fc_freq_5_NoTiedWeight_L1_Small
from models.fc_freq_Models import build_fc_freq_5_NoTiedWeight_L1_Tiny
from models.fc_freq_Models import build_fc_freq_5_TiedWeight_L1_Small
from models.fc_freq_Models import build_fc_freq_5_TiedWeight_L1_Tiny
from models.fc_freq_Models import build_fc_freq_5_TiedWeight_Big
from models.fc_freq_Models import build_fc_freq_5_NoTiedWeight_Big


from models.fc_freqComponent_Models import build_network_NoTiedWeight



from models.fc_freq_Models import build_fc_freq_4_30_NoTiedWeight_BN_Tiny

from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_BN_Contrib_Tiny
from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_BN_Wrapper_Tiny
from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_Behind_Tiny
from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_BN_Contrib_Behind_Tiny
from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_BN_Wrapper_Behind_Tiny

#from models.fc_freq_BN_Models import build_fc_freq_4_30_NoTiedWeight_BN_Contrib_Tiny
from models.fc_freq_BN_Models import build_fc_freq_4_30_NoTiedWeight_BN_Wrapper_Tiny
#from models.fc_freq_BN_Models import build_fc_freq_4_30_NoTiedWeight_BN_Contrib_Behind_Tiny
from models.fc_freq_BN_Models import build_fc_freq_4_30_NoTiedWeight_BN_Wrapper_Behind_Tiny
from models.fc_freq_BN_Models import build_fc_freqSum_NoTiedWeight_BN_Wrapper_Behind_Tiny


#from models.fc_freq_Models import build_fc_freq_5_NoTiedWeight_BN_Tiny

#
#def get_input_data_path(model, data_base_dir):
#	if model == "big":
#		# this is without time sampling
#		sub_volumes_dir = data_base_dir + '/' + 'sub_volumes_except_time'
#	elif model == "small":
#		# this is without time sampling
#		sub_volumes_dir = data_base_dir + '/' + 'sub_volumes_except_time'
#	elif model == "freqSumSmall":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSumBig":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_TiedWeight":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_TiedWeight_Big":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_NoTiedWeight_Big":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_NoTiedWeight_Small":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_NoTiedWeight_BN_Small":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freqSum_NoTiedWeight_Tiny":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freq_4_30_NoTiedWeight_Small":
#		# this is freq_4_30
#		sub_volumes_dir = '/data1/volumes_freq_4_30'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
#	elif model == "freq_4_30_NoTiedWeight_L1_Tiny":
#		# this is freq_4_30
#		sub_volumes_dir = '/data1/volumes_freq_4_30'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
#	elif model == "freq_4_30_NoTiedWeight_BN_Tiny":
#		# this is freq_4_30
#		#sub_volumes_dir = '/data1/volumes_freq_4_30'
#		sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
#	elif model == "freq_4_30_TiedWeight_Small":
#		# this is freq_4_30
#		sub_volumes_dir = '/data1/volumes_freq_4_30'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
#	elif model == "freq_5_TiedWeight_Small":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_4_30'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_Small":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_L1_Small":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_L1_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_TiedWeight_L1_Small":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_TiedWeight_L1_BN_Small":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_TiedWeight_L1_BN_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_BN_Contrib_Tiny":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_BN_Contrib_Behind_Tiny":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_BN_Wrapper_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freqSum_NoTiedWeight_BN_Wrapper_Behind_Tiny":
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	elif model == "freq_5_NoTiedWeight_BN_Wrapper_Behind_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_Behind_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_4_30_NoTiedWeight_BN_Wrapper_Behind_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_4_30'
#		sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
#	elif model == "freq_5_NoTiedWeight_L1_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_BN_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_TiedWeight_L1_Tiny":
#		# this is freq_5
#		#sub_volumes_dir = '/data1/volumes_freq_5'
#		sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_TiedWeight_Big":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freq_5_NoTiedWeight_Big":
#		# this is freq_5
#		sub_volumes_dir = '/data1/volumes_freq_5'
#		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
#	elif model == "freqSum_NoTiedWeight_Medium":
#		# this is freqSum 
#		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
#	else:
#		raise Exception("Invalid model name")
#
#	return sub_volumes_dir



def get_input_data_path(model, data_base_dir, is_local=True):
	if model == "freq_5_NoTiedWeight":
		if is_local is True:
			sub_volumes_dir = '/home/chuong/volumes_freq_5'
		else:
			sub_volumes_dir = '/data1/volumes_freq_5'
	elif model == "freq_4_30_NoTiedWeight":
		if is_local is True:
			sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
		else:
			sub_volumes_dir = '/data1/volumes_freq_4_30'
	elif model == "freqSum_NoTiedWeight":
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	else:
		raise Exception("Invalid model name")

	return sub_volumes_dir



def get_data_path_with_timestamp(model, data_base_dir):
	time_postfix = datetime.now().strftime('%Y-%m-%d-%H%M%S')
	return data_base_dir + '/' + model + '/' + time_postfix
