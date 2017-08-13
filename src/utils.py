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


from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_L1_BN_Small
from models.fc_freq_BN_Models import build_fc_freq_5_NoTiedWeight_L1_BN_Tiny

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
	elif model == "freqSum_NoTiedWeight_Big":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSum_NoTiedWeight_Small":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSum_NoTiedWeight_BN_Small":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freqSum_NoTiedWeight_Tiny":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	elif model == "freq_4_30_NoTiedWeight_Small":
		# this is freq_4_30
		sub_volumes_dir = '/data1/volumes_freq_4_30'
		#sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
	elif model == "freq_4_30_TiedWeight_Small":
		# this is freq_4_30
		sub_volumes_dir = '/data1/volumes_freq_4_30'
		#sub_volumes_dir = '/home/chuong/volumes_freq_4_30'
	elif model == "freq_5_TiedWeight_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_4_30'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_L1_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_L1_Tiny":
		# this is freq_5
		#sub_volumes_dir = '/data1/volumes_freq_5'
		sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_TiedWeight_L1_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_TiedWeight_L1_BN_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_TiedWeight_L1_BN_Tiny":
		# this is freq_5
		#sub_volumes_dir = '/data1/volumes_freq_5'
		sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_L1_BN_Small":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_L1_BN_Tiny":
		# this is freq_5
		#sub_volumes_dir = '/data1/volumes_freq_5'
		sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_L1_Tiny":
		# this is freq_5
		#sub_volumes_dir = '/data1/volumes_freq_5'
		sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_TiedWeight_L1_Tiny":
		# this is freq_5
		#sub_volumes_dir = '/data1/volumes_freq_5'
		sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_TiedWeight_Big":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freq_5_NoTiedWeight_Big":
		# this is freq_5
		sub_volumes_dir = '/data1/volumes_freq_5'
		#sub_volumes_dir = '/home/chuong/volumes_freq_5'
	elif model == "freqSum_NoTiedWeight_Medium":
		# this is freqSum 
		sub_volumes_dir = data_base_dir + '/' + 'volumes_freqSum'
	else:
		raise Exception("Invalid model name")

	return sub_volumes_dir


def build_model(model, x, x_dim, 
		dropout_keep_prob, gamma, 
		feature_activation, is_training):
	if model == 'big':
		print("Doing big model ...")
		loss, decoded = build_fc_big_freqFlatten(x, x_dim, dropout_keep_prob)
	elif model == 'freqSumSmall':
		print("Doing small model with freqSum model ...")
		loss, decoded = build_fc_freqFlatten_L1(x, x_dim, dropout_keep_prob)
	elif model == 'freqSum_TiedWeight_NoBias':
		loss, decoded, l1_loss = build_fc_freqSum_NoBias(
				x, x_dim, dropout_keep_prob, gamma)
	elif model == 'freqSum_TiedWeight':
		loss, decoded, l1_loss = build_fc_freqSum_TiedWeight(
			   x, x_dim, dropout_keep_prob, gamma)
	elif model == 'freqSum_TiedWeight_Big':
		loss, decoded, l1_loss = build_fc_freqSum_TiedWeight_Big(
			   x, x_dim, dropout_keep_prob,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freqSum_NoTiedWeight_Big':
		loss, decoded, l1_loss = build_fc_freqSum_NoTiedWeight_Big(
			   x, x_dim, dropout_keep_prob,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freqSum_NoTiedWeight_Tiny':
		loss, decoded, l1_loss = build_fc_freqSum_NoTiedWeight_Tiny(
			   x, x_dim, dropout_keep_prob, 
			   gamma=gamma, activation=feature_activation)
	elif model == 'freqSum_NoTiedWeight_Small':
		loss, decoded, l1_loss = build_fc_freqSum_NoTiedWeight_Small(
			   x, x_dim, dropout_keep_prob, 
			   gamma=gamma, activation=feature_activation)
	elif model == 'freqSum_NoTiedWeight_BN_Small':
		loss, decoded, l1_loss = build_fc_freqSum_NoTiedWeight_BN_Small(
			   x, x_dim, dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_4_30_NoTiedWeight_Small':
		loss, decoded, l1_loss = build_fc_freq_4_30_NoTiedWeight_Small(
			   x, x_dim, dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_4_30_TiedWeight_Small':
		loss, decoded, l1_loss = build_fc_freq_4_30_TiedWeight_Small(
			   x, x_dim, dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_TiedWeight_Small':
		loss, decoded, l1_loss = build_fc_freq_5_TiedWeight_Small(
			   x, x_dim, dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_TiedWeight_Big':
		loss, decoded, l1_loss = build_fc_freq_5_TiedWeight_Big(
			   x, x_dim, dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_Big':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_Big(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_Small':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_Small(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_L1_Small':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_L1_Small(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_L1_BN_Small':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_L1_BN_Small(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_L1_BN_Tiny':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_L1_BN_Tiny(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_NoTiedWeight_L1_Tiny':
		loss, decoded, l1_loss = build_fc_freq_5_NoTiedWeight_L1_Tiny(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_TiedWeight_L1_Tiny':
		loss, decoded, l1_loss = build_fc_freq_5_TiedWeight_L1_Tiny(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	elif model == 'freq_5_TiedWeight_L1_Small':
		loss, decoded, l1_loss = build_fc_freq_5_TiedWeight_L1_Small(
			   x, x_dim,  dropout_keep_prob, is_training,
			   gamma=gamma, activation=feature_activation)
	else:
		print("Doing small L1 model ...")
		loss, decoded = build_fc_freqSum_L1(x, x_dim, dropout_keep_prob, gamma)

	return loss, decoded, l1_loss


def get_data_path_with_timestamp(model, data_base_dir):
	time_postfix = datetime.now().strftime('%Y-%m-%d-%H%M%S')
	return data_base_dir + '/' + model + '/' + time_postfix
