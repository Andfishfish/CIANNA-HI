
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

load_epoch = 2000
if (len(sys.argv) > 1):
	load_epoch = int(sys.argv[1])

init_data_gen()
nb_ra_sky = 7
nb_dec_sky = 3
nb_area_freq = 50

input_test, targets_test = create_test_batch()
nb_test = nb_ra_sky*nb_dec_sky*nb_area_freq

if(1):

	cnn.init(in_dim=i_ar([sky_size,sky_size,freq_size]), in_nb_ch=1, 
			out_dim=1+max_nb_obj_per_image*(7+nb_param),
			bias=0.1, b_size=32, comp_meth='C_CUDA', dynamic_load=1, 
			mixed_precision="FP16C_FP32A", inference_only=1, adv_size=30)

	cnn.create_dataset("TEST", nb_test, input_test[:,:], targets_test[:,:])

	nb_yolo_filters = cnn.set_yolo_params(no_override = 0, raw_output = 0)

	cnn.load("/home/trzguo/content/SDC2/Cornu_et_al_2026_SDC2_models_catalogs_and_codes_archive/codes/complete_training_and_inference_pipeline/net_save/net0_s%04d.dat"%load_epoch,load_epoch, bin=1)

	#cnn.print_arch_tex("./arch/", "arch", activation=1, dropout=1)

	cnn.forward(saving=2, no_error=1)
	cnn.perf_eval()
	cnn.delete_dataset("TEST")

process_pred("fwd_res/net0_%04d.dat"%(load_epoch), "filtered_pred_ldev.txt")

assemble_and_build_catalog()

scoring()




