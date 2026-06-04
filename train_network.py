
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from threading import Thread
import subprocess
import sys, glob

#sys.path.insert(0,glob.glob('/path_to_CIANNA/src/build/lib.*/')[-1])
import CIANNA as cnn

from ldev_data_gen import *

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def data_augm():
	input_data, targets = create_train_batch()
	cnn.delete_dataset("TRAIN_buf", silent=1)
	cnn.create_dataset("TRAIN_buf", nb_images_per_iter, input_data[:,:], targets[:,:], silent=1)
	return

cnn.init(in_dim=i_ar([sky_size,sky_size,freq_size]), in_nb_ch=1, out_dim=1+max_nb_obj_per_image*(7+nb_param+diff_flagging),
	bias=0.1, b_size=16, comp_meth='C_CUDA', dynamic_load=1, mixed_precision="FP16C_FP32A", adv_size=20)

init_data_gen()

input_data, targets = create_train_batch()
input_valid, targets_valid = create_valid_batch()

cnn.create_dataset("TRAIN", nb_images_per_iter, input_data[:,:], targets[:,:])
cnn.create_dataset("VALID", nb_valid, input_valid[:,:], targets_valid[:,:])

##### YOLO parameters tuning #####

#Size priors for all possible boxes per grid. element
prior_size = f_ar([[10.0], [10.0], [10.00]]) #原本频率方向是36，但是alfalfa这边的数据不知道为什么源的频率宽度都是10(可能和我pa==180,i==45有关)

#No obj probability prior to rebalance the size distribution
#prior_noobj_prob = f_ar([0.0004])
prior_noobj_prob = f_ar([0.01])

#Relative scaling of each error "type" : 
error_scales = cnn.set_error_scales(position = 2.0, size = 1.0, probability = 1.0, objectness = 6.0, parameters = 3.0)

#Relative scaling of each extra paramater
param_ind_scales = f_ar([3.0,2.0,2.0,1.0,1.0,2.0])

#Various IoU limit conditions
IoU_limits = cnn.set_IoU_limits(
	good_IoU_lim = 0.4, min_prob_IoU_lim = -0.1, 
	min_obj_IoU_lim = -0.1,	min_param_IoU_lim = 0.4, 
	diff_IoU_lim = 0.1, diff_obj_lim = 0.2)

#Activate / deactivate some parts of the loss
fit_parts = cnn.set_fit_parts(position = 1, size = 1, probability = 1, objectness = 1, parameters = 1)

#Supplementary parameters for activation function of each part
slopes_and_maxes = cnn.set_slopes_and_maxes(
	position    = cnn.set_sm_single(slope = 0.5, fmax = 8.0, fmin = -8.0),
	size        = cnn.set_sm_single(slope = 0.5, fmax = 1.8, fmin = -1.4),
	probability = cnn.set_sm_single(slope = 0.2, fmax = 8.0, fmin = -8.0),
	objectness  = cnn.set_sm_single(slope = 0.5, fmax = 8.0, fmin = -8.0),
	parameters  = cnn.set_sm_single(slope = 0.5, fmax = 1.2, fmin = -0.2))

strict_box_size = 0 #have no impact with only a single box prior

nb_yolo_filters = cnn.set_yolo_params(nb_box = nb_box, nb_class = 0, nb_param = nb_param, max_nb_obj_per_image = max_nb_obj_per_image,
				prior_size = prior_size, prior_noobj_prob = prior_noobj_prob, IoU_type = "DIoU", prior_dist_type = "OFFSET",
				error_scales = error_scales, param_ind_scales = param_ind_scales, slopes_and_maxes = slopes_and_maxes, IoU_limits = IoU_limits,
				fit_parts = fit_parts, strict_box_size = strict_box_size, diff_flag=diff_flagging, rand_startup = 0, error_type = "natural", no_override = 1, raw_output = 0)

a_relu = cnn.relu(leaking=0.1, saturation=640000.0)

load_epoch = 3950
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

if(load_epoch > 0):
	cnn.load("net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
else:

	cnn.conv(f_size=i_ar([1,1,8]), nb_filters=12 , stride=i_ar([1,1,2]), padding=i_ar([0,0,3]), int_padding=i_ar([0,0,0]), activation=a_relu)
	#64x64x128
	cnn.conv(f_size=i_ar([3,3,1]), nb_filters=8  , stride=i_ar([1,1,1]), padding=i_ar([1,1,0]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,5]), nb_filters=8  , stride=i_ar([1,1,1]), padding=i_ar([0,0,2]), int_padding=i_ar([0,0,0]), activation=a_relu)
	
	cnn.conv(f_size=i_ar([2,2,6]), nb_filters=8  , stride=i_ar([2,2,2]), padding=i_ar([0,0,2]), int_padding=i_ar([0,0,0]), activation=a_relu)
	#32x32x64
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=16 , stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=12 , stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)

	cnn.conv(f_size=i_ar([1,1,6]), nb_filters=16 , stride=i_ar([1,1,2]), padding=i_ar([0,0,2]), int_padding=i_ar([0,0,0]), activation=a_relu)
	#32x32x32
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=24 , stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=16 , stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=24 , stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=16 , stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	
	cnn.conv(f_size=i_ar([2,2,3]), nb_filters=48 , stride=i_ar([2,2,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	#16x16x32
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=96 , stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=48 , stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=96 , stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=48 , stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)

	cnn.conv(f_size=i_ar([2,2,4]), nb_filters=128, stride=i_ar([2,2,2]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	#8x8x16
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=256, stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,3]), nb_filters=128, stride=i_ar([1,1,1]), padding=i_ar([0,0,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([3,3,3]), nb_filters=256, stride=i_ar([1,1,1]), padding=i_ar([1,1,1]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.norm(group_size=2)
	cnn.conv(f_size=i_ar([1,1,1]), nb_filters=1024, stride=i_ar([1,1,1]), padding=i_ar([0,0,0]), int_padding=i_ar([0,0,0]), activation=a_relu, drop_rate = 0.25)
	cnn.conv(f_size=i_ar([1,1,1]), nb_filters=768 , stride=i_ar([1,1,1]), padding=i_ar([0,0,0]), int_padding=i_ar([0,0,0]), activation=a_relu)
	cnn.conv(f_size=i_ar([1,1,1]), nb_filters=nb_yolo_filters, stride=i_ar([1,1,1]), padding=i_ar([0,0,0]), int_padding=i_ar([0,0,0]), activation="YOLO")


#cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=1)

learning_rate = 0.0006
max_epoch = 8000

for block in range(load_epoch,max_epoch):
	
	t = Thread(target=data_augm)
	t.start()
	
	n_startup = 20
	
	if((block+1) <= n_startup):
		loc_lr = 0.98*learning_rate*((block+1)/n_startup)+0.02*learning_rate
	else:
		loc_lr = learning_rate
	
	cnn.train(nb_iter=1, learning_rate=loc_lr, end_learning_rate=loc_lr*0.02, shuffle_every=0,\
				 momentum=0.7, lr_decay=0.0005, weight_decay=0.0005, save_every=50, silent=0, save_bin=1, TC_scale_factor=64.0)
	
	if(block == 0):
		cnn.perf_eval()

	t.join()
	
	cnn.swap_data_buffers("TRAIN")

















