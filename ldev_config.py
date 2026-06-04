
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from aux_fct import *

######	  GLOBAL VARIABLES AND DATA	  #####
# Data are expected to be square in the RA and DEC axis  
ra_pixel_size = 486
dec_pixel_size = 160
map_pixel_freq_size = 13016
pixel_size = 1.6666666667E-02 #In degree
pixel_size_freq = 7629.39453125 #In Hz
beam_size = 6.00925212577E-03 #In degree

work_path = "/home/trzguo/content/github"

#Normalization parameters 
do_norm = 0
freq_smoothing = 0
kernel_size = 20
cube_spliting = 0
size_file_split_limit = 4 #In Gigabytes
compute_std_norm = 1

cont_removal_threshold = 0.5*6e-3 #in Jy
prenorm_scaling = 0.3



#####    NETWORK RELATED GLOBAL VARIABLES     #####
sky_size = 64
freq_size = 256
nb_param = 6
nb_box = 1
diff_flagging = 1

#####    TRAINING RELATED GLOBAL VARIABLES    #####

flux_clip = [0,300.0] #由于alfalfa中的源流量较小所以下限改为0
hi_size_clip = [20.0, 700.0]
w20_clip = [100.0, 700.0]
angle_res_lim = 6

nb_images_per_iter = 1600
nb_valid = 300
max_nb_obj_per_image = 100 #10 因为有些patch不知道为什么能截出二十多个源
bootstrap = 0

sure_source_frac = 0.3 #can be set higher for pre-training -> noobj factor must be scaled accordingly
add_border_frac = 0.2 #Chance of presence for added Null border in each axis (only one side at a time)

flip_hor  = 0.5
flip_vert = 0.5
flip_freq = 0.5


#####   INFERENCE RELATED GLOBAL VARIABLES    #####

c_size_sky = 8
c_size_freq = 16
yolo_nb_sky_reg = int(sky_size/c_size_sky)
yolo_nb_freq_reg = int(freq_size/c_size_freq)

overlap_sky = c_size_sky
overlap_freq = 2*c_size_freq
patch_shift_sky = sky_size - overlap_sky
patch_shift_freq = freq_size - overlap_freq

orig_offset_ra = patch_shift_sky - ((int(ra_pixel_size/2) - int(sky_size/2) + patch_shift_sky)%patch_shift_sky)
orig_offset_dec = patch_shift_sky - ((int(dec_pixel_size/2) - int(sky_size/2) + patch_shift_sky)%patch_shift_sky)
orig_offset_freq = patch_shift_freq - ((int(map_pixel_freq_size/2) - int(freq_size/2) + patch_shift_freq)%patch_shift_freq)

nb_area_ra = int((ra_pixel_size+2*orig_offset_ra)/patch_shift_sky)
nb_area_dec = int((dec_pixel_size+2*orig_offset_dec)/patch_shift_sky)
nb_area_freq = int((map_pixel_freq_size+2*orig_offset_freq)/patch_shift_freq)

# Function is duplicated in each config file for simpler reading of all above parameters
# Expect fits files without the extension
def cube_norm(cube_path, name_prefix):
	
	hdul = fits.open(cube_path, memmap=True)
	cube_data = hdul[0].data
	cube_data = np.transpose(cube_data, (0, 2, 1))
	
	# Remove very bright pixels from the continuum in the cube due to residual substraction errors
	# index = np.where(np.mean(np.asarray(continuum_data,dtype="float32"),axis=0) > 0.5*6e-3)
	
	if(compute_std_norm):
		c_norm = np.zeros(np.shape(cube_data)[0])
		for i in range(0, np.shape(cube_data)[0]):
			cube_slice = np.asarray(cube_data[i], dtype="float32")
			#cube_slice[index] = 0.0
			c_norm[i] = np.nanstd(cube_slice, axis=(0,1))
		np.savetxt(name_prefix+"_c_norm.dat", c_norm)
	else: # For the MAIN/FULL cube, it is often better to use the std estimated on the LDEV as well (especially if trained on the LDEV)
		c_norm = np.loadtxt("LDEV_c_norm.dat")
	
	# Split the cube into chuncks that can be saved as files smaller than size_file_split_limit using uint16 encoding
	if(cube_spliting):
		encoding_bytes = 2
		nb_sky_patch_per_subcube = int((np.sqrt(size_file_split_limit*1e9/(encoding_bytes*(map_pixel_freq_size+2*orig_offset_freq))) - overlap_sky)/patch_shift_sky)
		sub_cube_pixel_size = nb_sky_patch_per_subcube*patch_shift_sky + overlap_sky
		nb_sub_cube_per_dim = int(np.ceil((map_pixel_size + orig_offset_sky*2)/sub_cube_pixel_size))
	
	# If not spliting, the number of chunks is one
	# Works better for the LDEV that should be fully loaded for efficient dynamical augmentation during training
	else:
		nb_ra_patch_per_subcube = nb_area_ra
		nb_dec_patch_per_subcube = nb_area_dec
		sub_ra_pixel_size = nb_ra_patch_per_subcube*patch_shift_sky + overlap_sky #176
		sub_dec_pixel_size = nb_dec_patch_per_subcube*patch_shift_sky + overlap_sky #176
		print("Sub-ra pixel size: %d"%(sub_ra_pixel_size))
		print("Sub-dec pixel size: %d"%(sub_dec_pixel_size))
		nb_sub_cube_per_dim = 1
	
	
	# Tried parallelization on chunck loading but it reduced performances, strongly I/O limited
	for n_h in tqdm(range(0,nb_sub_cube_per_dim)):
		for n_w in range(0,nb_sub_cube_per_dim):
	
			norm_data = np.zeros((map_pixel_freq_size+2*orig_offset_freq,
				min(sub_ra_pixel_size, (ra_pixel_size + orig_offset_ra*2 - n_h*(sub_ra_pixel_size-overlap_sky))), 
				min(sub_dec_pixel_size, (dec_pixel_size + orig_offset_dec*2 - n_w*(sub_dec_pixel_size-overlap_sky)))), dtype="uint16")
			
			print(min(sub_ra_pixel_size, (ra_pixel_size + orig_offset_ra*2 - n_h*(sub_ra_pixel_size-overlap_sky))))
			print(min(sub_dec_pixel_size, (dec_pixel_size + orig_offset_dec*2 - n_w*(sub_dec_pixel_size-overlap_sky))))
			y_min = n_h*nb_ra_patch_per_subcube*patch_shift_sky - orig_offset_ra
			y_max = (n_h+1)*nb_ra_patch_per_subcube*patch_shift_sky + overlap_sky - orig_offset_ra
			x_min = n_w*nb_dec_patch_per_subcube*patch_shift_sky - orig_offset_dec
			x_max = (n_w+1)*nb_dec_patch_per_subcube*patch_shift_sky + overlap_sky - orig_offset_dec
			
			orig_y_min = max(0, y_min)
			orig_y_max = min(ra_pixel_size, y_max)
			orig_x_min = max(0, x_min)
			orig_x_max = min(dec_pixel_size, x_max)
			
			data = np.asarray(cube_data[:,orig_y_min:orig_y_max,orig_x_min:orig_x_max], dtype="float32")
			#continuum = np.mean(np.asarray(continuum_data[:,orig_y_min:orig_y_max,orig_x_min:orig_x_max], dtype="float32"), axis=0)
			
			# Remove very bright pixels from the continuum in the cube due to residual substraction errors
			# index = np.where(continuum > 0.5*6e-3)
			# data[:,index[0][:],index[1][:]] = 0.0
			
			# If required, smooth the frequency axis
			'''
			if(freq_smoothing):
				kernel = np.zeros((kernel_size)) + 1.0
				conv = np.zeros((map_pixel_freq_size))
				for k in tqdm(range(0, map_pixel_size)):
					for l in range(0, map_pixel_size):
						conv[:] = signal.convolve(data[:,k,l], kernel, mode="same")
						data[:,k,l] = conv[:]
			'''

			# Renormalize by each channel noise, scale, and tanh transform, them remap in [0, 1]
			for i in range(0, np.shape(data)[0]):
				slice_data = data[i,:,:]
				# 临时填充NaN为0，不影响有效数据的归一化（因c_norm已用nanstd计算）
				slice_data_nonan = np.nan_to_num(slice_data, nan=0.0)
				# tanh变换 + 映射到[0,1]
				normed_slice = (np.tanh(prenorm_scaling * slice_data_nonan / c_norm[i]) + 1.0) * 0.5
				# 还原NaN位置（后续填充为中性值）
				normed_slice[np.isnan(slice_data)] = np.nan
				data[i,:,:] = normed_slice
				#data[i,:,:] = (np.tanh(prenorm_scaling*data[i,:,:] / c_norm[i]) + 1.0)*0.5
			
			np.savetxt(work_path+name_prefix+"_c_norm.dat", c_norm)

			end_y_min = max(0, -y_min)
			end_y_max = sub_ra_pixel_size - max(0, y_max - ra_pixel_size) 
			end_x_min = max(0, -x_min)
			end_x_max = sub_dec_pixel_size - max(0, x_max - dec_pixel_size) 

			# Scale to fit in uint16 format and save a binary file
			norm_data[:,:,:] = 0.5*65535.0
			norm_data[orig_offset_freq:-orig_offset_freq,end_y_min:end_y_max,end_x_min:end_x_max] = np.asarray(data[:,:,:] * 65535.0, dtype="uint16")
			print(f"写入的数组维度: {norm_data.shape}")
			print(f"写入的元素数: {norm_data.size}")
			norm_data.tofile(work_path+name_prefix+"_%d_%d.bin"%(n_h, n_w))

			del (norm_data)
			gc.collect()



