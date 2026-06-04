
#	Author and copyright (C) 2026 - David Cornu
#	Code associated with the article Cornu et al. 2026 (A&A)
#	Released as part of the archived deposit 10.5281/zenodo.18403011

from ldev_config import *

def init_data_gen():

	global norm_data, select_cat, flux_list, hi_size_list, w20_list, pa_list, inc_list, diff_list
	global coords, x, y, z, w, h, d
	global input_data, targets, input_valid, targets_valid

	if(do_norm):
		print("Normalizing LDEV cube, this will take a while ...")
		cube_norm("merged_cube.fits", "LDEV") 
			#work_path, kernel_size, cube_spliting, size_file_split_limit, compute_std_norm)
	
	hdul = fits.open("merged_cube.fits", memmap=True)
	wcs_cube = WCS(hdul[0].header)
	
	l_ra_pixel_size = ra_pixel_size + orig_offset_ra*2
	l_dec_pixel_size = dec_pixel_size + orig_offset_dec*2
	l_map_pixel_freq_size = map_pixel_freq_size+2*orig_offset_freq
	
	norm_data = np.fromfile(work_path+"LDEV_0_0.bin", dtype="uint16")
	norm_data = np.reshape(norm_data, (l_map_pixel_freq_size, l_ra_pixel_size, l_dec_pixel_size))
	
	# id, ra, dec, hi_size, line_flux_integral, central_freq, pa, i, w20
	true_cat = np.loadtxt("alfalfa_full_catalog.txt", skiprows=1)
	print("Orig true_cat size", np.shape(true_cat))

	# 转换天球坐标 → 像素坐标
	c = SkyCoord(ra=true_cat[:,1]*u.degree, dec=true_cat[:,2]*u.degree, frame='icrs')
	x, y = utils.skycoord_to_pixel(c, wcs_cube, origin=0)
	z = (1420.40575177/(1 + true_cat[:,5]/299792.458) - 1320.003271102905)/pixel_size_freq * 1e6 + orig_offset_freq + 0.5

	# ==============================================
	# 🔥 关键：过滤掉 超出FITS图像范围 / NaN 的源
	# ==============================================
	valid_mask = ~np.isnan(x) & ~np.isnan(y)  # 不是 NaN
	valid_mask &= (x >= 0) & (x < l_ra_pixel_size)  # 在图像 X 范围内
	valid_mask &= (y >= 0) & (y < l_dec_pixel_size)  # 在图像 Y 范围内
	valid_mask &= (z >= 0) & (z < l_map_pixel_freq_size)  # 在图像 Z 范围内

	# 只保留有效源
	true_cat = true_cat[valid_mask]
	x = x[valid_mask]
	y = y[valid_mask]
	z = z[valid_mask]

	print(f"✅ 保留在图像内的源数量: {np.sum(valid_mask)} / {len(valid_mask)}")
	print(f"❌ 剔除超出范围的源数量: {len(valid_mask) - np.sum(valid_mask)}")

	width  = np.sqrt(((true_cat[:,3]/3600)**2+beam_size**2))/pixel_size
	height = width
	depth  = (true_cat[:,8]/299792.458)*(true_cat[:,5]**2/1.4204e9)/pixel_size_freq

	line_flux = true_cat[:,4]
	
	source_volume = np.pi * width**2 * depth
	
	if(not bootstrap): #First order selection
		index_keep = np.where((line_flux > 1.0))[0]
		select_cat = true_cat[index_keep]
		print ("Nb selected by naiv fct:", np.shape(select_cat)[0])
	else: #Second order selection
		index_keep = np.where((line_flux > 1.0))[0]
		select_cat = true_cat[index_keep]
		print ("Nb selected by naiv fct:", np.shape(select_cat)[0])
	
	# Recompute for selected sources only
	c_sel = SkyCoord(ra=select_cat[:,1]*u.degree, dec=select_cat[:,2]*u.degree, frame='icrs')
	x_sel, y_sel = utils.skycoord_to_pixel(c_sel, wcs_cube, origin=0)
	x = x_sel + orig_offset_ra + 0.5
	y = y_sel + orig_offset_dec + 0.5
	z = (1420.40575177/(1 + select_cat[:,5]/299792.458) - 1320.003271102905)/pixel_size_freq * 1e6 + orig_offset_freq + 0.5
	
	width  = np.sqrt(((select_cat[:,3]/3600)**2+beam_size**2))/pixel_size
	height = width
	depth  = (select_cat[:,8]/299792.458)*(select_cat[:,5]**2/1.4204e9)/pixel_size_freq
	
	line_flux = select_cat[:,4]
	
	source_volume = np.pi * width**2 * depth
	
	pa = select_cat[:,6]
	
	coords = np.zeros((np.shape(select_cat)[0],6))
	
	for i in range(0,np.shape(width)[0]):
		vertices = np.array([[-width[i]*0.5,-height[i]*0.5],[-width[i]*0.5,height[i]*0.5],\
		                     [width[i]*0.5, -height[i]*0.5],[width[i]*0.5,height[i]*0.5]])
		
		vertices_new = np.zeros((4,2))
		vertices_new[:,0] = np.cos(pa[i]*np.pi/180.0)*vertices[:,0] + np.sin(pa[i]*np.pi/180.0)*vertices[:,1]
		vertices_new[:,1] = - np.sin(pa[i]*np.pi/180.0)*vertices[:,0] + np.cos(pa[i]*np.pi/180.0)*vertices[:,1]
		
		coords[i,0] = min(vertices_new[:,0]) - 2
		coords[i,1] = max(vertices_new[:,0]) + 2
		coords[i,2] = min(vertices_new[:,1]) - 2
		coords[i,3] = max(vertices_new[:,1]) + 2
		coords[i,4] = -0.5*depth[i] - 5
		coords[i,5] = +0.5*depth[i] + 5
		print(f"源{i}: RA框宽={coords[i,1]-coords[i,0]:.1f},DEC={coords[i,3]-coords[i,2]:.1f},FREQ={coords[i,5]-coords[i,4]:.1f}")
	
	coords[:,0] += x; coords[:,1] += x
	coords[:,2] += y; coords[:,3] += y
	coords[:,4] += z; coords[:,5] += z
	
	w = np.abs(coords[:,0]-coords[:,1])
	h = np.abs(coords[:,2]-coords[:,3])
	d = np.abs(coords[:,4]-coords[:,5])
	
	if(bootstrap):
		predet_cat = np.loadtxt("filtered_pred_ldev_in.txt")
		print ("Nb of previous detections:", np.shape(predet_cat)[0])
		max_prob_sequence = 1600
		DIoU_match_limit = 0.1
		
		match_score = np.zeros((np.shape(select_cat)[0]))
		
		for k in range(0, np.shape(select_cat)[0]):
			l_coords = coords[k,[0,2,4,1,3,5]]
			best_f_IoU = -1.0
			for l in range(0, min(max_prob_sequence,np.shape(predet_cat)[0])):
				f_IoU = fct_DIoU(l_coords, predet_cat[l,0:6])
				if(f_IoU > DIoU_match_limit and f_IoU > best_f_IoU):
					best_f_IoU = f_IoU
					match_score[k] = predet_cat[l,7]
		
		index = np.where((match_score > 0.1) | ((line_flux > 100.0) | ((line_flux > 18.0) & (line_flux/source_volume > 0.013))))[0]
		
		print("Nb of selected X predet matches", np.shape(index))
		print("Min-max objectness selected:", np.min(match_score[index]), np.max(match_score[index]))
		
		select_cat = select_cat[index]
		coords = coords[index]
		w = w[index]; h = h[index]; d = d[index]
		
		if(diff_flagging):
			#Add difficult flag for all examples from the original selection that does not match 
			# and new sources added from match outside the original selection
			match_score = match_score[index]
			line_flux = line_flux[index]
			source_volume = source_volume[index]
			diff_list = np.zeros(np.shape(select_cat)[0]) + 1
			
			index_diff = np.where((line_flux > 100.0) | ((line_flux > 18.0) & (line_flux/source_volume > 0.013)))[0]
			diff_list[index_diff] = 0
			
			print ("Number of sources flagged as difficut:", np.sum(diff_list))
	
	flux_list = np.copy(select_cat[:,4])
	hi_size_list = np.copy(select_cat[:,3])
	w20_list = np.copy(select_cat[:,8])
	pa_list = np.copy(select_cat[:,6])
	inc_list = np.copy(select_cat[:,7])
	

	# Cap the minimum and maximum value for the Line flux, HI size and w20 for the regression 
	# (not linked to the bounding boxes definition)
	flux_list = np.clip(flux_list, flux_clip[0], flux_clip[1])
	hi_size_list = np.clip(hi_size_list, hi_size_clip[0], hi_size_clip[1])
	w20_list = np.clip(w20_list, w20_clip[0], w20_clip[1])
	
	w20_list = (w20_list/299792.458)*(select_cat[:,5]**2/1.4204e9)/pixel_size_freq
	
	# Flag small objects (hi or w20 size) for which PA or Inclination estimation is too difficult and set their target to a fixed median value
	small_id = np.where(hi_size_list < angle_res_lim)[0]
	pa_list[small_id] = 180.0
	# inc_list[small_id] = 45.0
	
	# Switch to logscale for line flux and w20 to obtain flatter distributions across scales
	flux_list = np.log(flux_list)
	w20_list = np.log(w20_list)
	
	# Get the normalization limits and save them to convert network predictions back to physical quantities
	flux_min = np.min(flux_list); flux_list -= flux_min;
	flux_max = np.max(flux_list); flux_list /= flux_max
	
	hi_size_min = np.min(hi_size_list); hi_size_list -= hi_size_min
	hi_size_max = np.max(hi_size_list)
	hi_size_list /= hi_size_max
	
	w20_min = np.min(w20_list); w20_list -= w20_min
	w20_max = np.max(w20_list); w20_list /= w20_max
	
	lims = np.zeros((3,2))
	lims[0] = [flux_max, flux_min]; lims[1] = [hi_size_max, hi_size_min]; lims[2] = [w20_max, w20_min]
	
	print ("\nMin-Max values used for normalization (Line flux, HI size, w20):")
	print (lims[0], lims[1], lims[2])
	
	np.savetxt("train_cat_lims.txt", lims)
	
	input_data = np.zeros((nb_images_per_iter,1*sky_size*sky_size*freq_size), dtype="float32")
	targets = np.zeros((nb_images_per_iter,1+max_nb_obj_per_image*(7+nb_param+diff_flagging)), dtype="float32")
	
	input_valid = np.zeros((nb_valid,1*sky_size*sky_size*freq_size), dtype="float32")
	targets_valid = np.zeros((nb_valid,1+max_nb_obj_per_image*(7+nb_param+diff_flagging)), dtype="float32")


def create_train_batch(visual=0):
	
	for i in range(0,nb_images_per_iter):
		
		rand_id = np.random.random()
		if(rand_id < sure_source_frac):
			
			s_id = int(np.random.randint(0, np.shape(select_cat)[0]))
			p_ra   = int(( x[s_id] - np.random.randint(w[s_id]*0.5+1, sky_size-w[s_id]*0.5-1)))
			p_dec  = int(( y[s_id] - np.random.randint(h[s_id]*0.5+1, sky_size-h[s_id]*0.5-1)))
			p_freq = int(( z[s_id] - np.random.randint(d[s_id]*0.5+1, freq_size-d[s_id]*0.5-1)))
			
			p_ra   = np.clip(p_ra,0,ra_pixel_size+orig_offset_ra*2-sky_size)
			p_dec  = np.clip(p_dec,0,dec_pixel_size+orig_offset_dec*2-sky_size)
			p_freq = np.clip(p_freq,0,map_pixel_freq_size+orig_offset_freq*2-freq_size)
			#print(f"→ 优先选源：s_id={s_id} | 源中心 RA={x[s_id]} DEC={y[s_id]} FREQ={z[s_id]} | 尺寸 w={w[s_id]} h={h[s_id]} d={d[s_id]}")
			
		else:
			p_ra   = np.random.randint(0,ra_pixel_size+orig_offset_ra*2-sky_size)
			p_dec  = np.random.randint(0,dec_pixel_size+orig_offset_dec*2-sky_size)
			p_freq = np.random.randint(0,map_pixel_freq_size+orig_offset_freq*2-freq_size)
		
		patch = (np.copy(norm_data[p_freq:p_freq+freq_size, p_ra:p_ra+sky_size, p_dec:p_dec+sky_size])/65535.0)*2.0 - 1.0
		
		cat_id = np.where((coords[:,0] >= p_ra)   & (coords[:,1] <= p_ra+sky_size) &\
						  (coords[:,2] >= p_dec)  & (coords[:,3] <= p_dec+sky_size) &\
						  (coords[:,4] >= p_freq) & (coords[:,5] <= p_freq+freq_size))[0]
		patch_ra = (p_ra, p_ra+sky_size)
		patch_dec = (p_dec, p_dec+sky_size)
		patch_freq = (p_freq, p_freq+freq_size)
		#print(f"Patch {i} | 裁剪范围 RA:{patch_ra} | DEC:{patch_dec} | FREQ:{patch_freq} | 源数量:{len(cat_id)}")
		#print(f"Patch {i:3d} | 源数量: {len(cat_id):2d} | {'✅ 有源' if len(cat_id) else '❌ 无源'}")
		coords
		axis_size = np.array((sky_size, sky_size, freq_size))
		
		# Add possible border in each axis
		mask_offset = 0.0 # np.random.normal(0,0.02) # Possible jitter for the masked pixel value
		for l in range(0,3):
			if(np.random.random() < add_border_frac):
				border_dir = np.random.randint(0,2)*2 - 1 # generate 1 or -1
				if(np.shape(cat_id)[0] > 0):
					l_coords = np.copy(coords[cat_id,:])
					l_coords[:,0:2] -= p_ra
					l_coords[:,2:4] -= p_dec
					l_coords[:,4:6] -= p_freq
					lim_axis_no_source = int(min(border_dir*l_coords[:,l*2:l*2+2].flatten()))
				else:
					lim_axis_no_source = border_dir*axis_size[l]//2
				
				slices = [slice(None)] * 3
				if border_dir == 1:
					amp = lim_axis_no_source+1
					slices[2-l] = slice(0, min(axis_size[l]//2,int(amp*np.random.random())))
				else:
					amp = -(lim_axis_no_source-1)
					slices[2-l] = slice(max(axis_size[l]//2,int(amp+(axis_size[l]-amp)*np.random.random())), None)
				
				patch[tuple(slices)] = mask_offset
		
		flip_w = 0; flip_h = 0; flip_d = 0
		if(np.random.random() < flip_hor):
			flip_w = 1
			patch = np.flip(patch, axis=1)
			
		if(np.random.random() < flip_vert):
			flip_h = 1
			patch = np.flip(patch, axis=2)
			
		if(np.random.random() < flip_freq):
			flip_d = 1
			patch = np.flip(patch, axis=0)
		input_data[i,0:freq_size*sky_size*sky_size] = patch.flatten("C")
		
		targets[i,:] = 0.0
		targets[i,0] = min(max_nb_obj_per_image, np.shape(cat_id)[0])
		if(np.shape(cat_id)[0] >= max_nb_obj_per_image):
			print ("Max obj. reached")
		
		for k in range(0, int(targets[i,0])):
			l_t_pa = np.copy(pa_list[cat_id[k]])
		
			l_coords = np.copy(coords[cat_id[k],:])
			l_coords[0:2] -= p_ra
			l_coords[2:4] -= p_dec
			l_coords[4:6] -= p_freq
			
			if(flip_w):
				l_t_pa = np.mod(-l_t_pa,360.0)
				xmax, xmin = sky_size - l_coords[0:2]
				l_coords[0:2] = (xmin, xmax)
			if(flip_h):
				l_t_pa = np.mod(180.0-l_t_pa,360.0)
				ymax, ymin = sky_size - l_coords[2:4]
				l_coords[2:4] = (ymin, ymax)
			if(flip_d):
				l_t_pa = np.mod(l_t_pa-180.0,360.0)
				zmax, zmin = freq_size - l_coords[4:6]
				l_coords[4:6] = (zmin, zmax)
			
			targets[i,1+k*(7+nb_param+diff_flagging)] = 1.0 # target class
			targets[i,1+k*(7+nb_param+diff_flagging)+1:1+k*(7+nb_param+diff_flagging)+7] = l_coords[[0,2,4,1,3,5]] # reordered boxes coords
			targets[i,1+k*(7+nb_param+diff_flagging)+7:1+k*(7+nb_param+diff_flagging)+7+nb_param] = \
				np.array([flux_list[cat_id[k]],\
						  hi_size_list[cat_id[k]],\
						  w20_list[cat_id[k]],
						  (np.sin(l_t_pa*np.pi/180.0)+1)*0.5,\
						  (np.cos(l_t_pa*np.pi/180.0)+1)*0.5,\
						  np.cos(inc_list[cat_id[k]]*np.pi/180.0)])
			if(diff_flagging and bootstrap):
				targets[i,1+k*(7+nb_param+diff_flagging)+7+nb_param] = diff_list[cat_id[k]]
	
	return input_data, targets


def create_valid_batch(visual=0):
	
	for i in range(0,nb_valid):
		s_id = i
		
		p_ra   = int( x[s_id] - sky_size-w[s_id]*0.5)
		p_dec  = int( y[s_id] - sky_size-h[s_id]*0.5)
		p_freq = int( z[s_id] - freq_size-d[s_id]*0.5)
		
		p_ra   = np.clip(p_ra,0,ra_pixel_size+orig_offset_ra*2-sky_size)
		p_dec  = np.clip(p_dec,0,dec_pixel_size+orig_offset_dec*2-sky_size)
		p_freq = np.clip(p_freq,0,map_pixel_freq_size+orig_offset_freq*2-freq_size)

		patch = np.copy(norm_data[p_freq:p_freq+freq_size, p_ra:p_ra+sky_size, p_dec:p_dec+sky_size])

		input_valid[i,0:freq_size*sky_size*sky_size] = (patch.flatten("C")/65535.0)*2.0 - 1.0
		
		cat_id = np.where((coords[:,0] >= p_ra)   & (coords[:,1] <= p_ra+sky_size) &\
						  (coords[:,2] >= p_dec)  & (coords[:,3] <= p_dec+sky_size) &\
						  (coords[:,4] >= p_freq) & (coords[:,5] <= p_freq+freq_size))[0]
		
		targets_valid[i,:] = 0.0
		targets_valid[i,0] = min(max_nb_obj_per_image, np.shape(cat_id)[0])
		if(np.shape(cat_id)[0] >= max_nb_obj_per_image):
			print ("Max obj. reached")
		
		for k in range(0, int(targets_valid[i,0])):
			l_t_pa = np.copy(pa_list[cat_id[k]])
		
			l_coords = np.copy(coords[cat_id[k],:])
			l_coords[0:2] -= p_ra
			l_coords[2:4] -= p_dec
			l_coords[4:6] -= p_freq
			
			targets_valid[i,1+k*(7+nb_param+diff_flagging)] = 1.0 #target class
			targets_valid[i,1+k*(7+nb_param+diff_flagging)+1:1+k*(7+nb_param+diff_flagging)+7] = l_coords[[0,2,4,1,3,5]] #reordered boxes coords
			targets_valid[i,1+k*(7+nb_param+diff_flagging)+7:1+k*(7+nb_param+diff_flagging)+7+nb_param] = \
				np.array([flux_list[cat_id[k]],\
						  hi_size_list[cat_id[k]],\
						  w20_list[cat_id[k]],
						  (np.sin(l_t_pa*np.pi/180.0)+1)*0.5,\
						  (np.cos(l_t_pa*np.pi/180.0)+1)*0.5,\
						  np.cos(inc_list[cat_id[k]]*np.pi/180.0)])
			if(diff_flagging and bootstrap):
				targets_valid[i,1+k*(7+nb_param+diff_flagging)+7+nb_param] = diff_list[cat_id[k]]
	
	return input_valid, targets_valid
	

def create_test_batch():
	
	nb_test = nb_area_sky*nb_area_sky*nb_area_freq
	
	input_test = np.zeros((nb_test,1*sky_size*sky_size*freq_size), dtype="float32")
	targets_test = np.zeros((nb_test,1+max_nb_obj_per_image*(7+nb_param+diff_flagging)), dtype="float32")
	
	for patch_freq in range(0,nb_area_freq):
		for patch_dec in range(0,nb_area_sky):
			for patch_ra in range(0,nb_area_sky):
				
				i = patch_freq*nb_area_sky*nb_area_sky + patch_dec*nb_area_sky + patch_ra
				
				p_ra   = patch_ra*patch_shift_sky
				p_dec  = patch_dec*patch_shift_sky
				p_freq = patch_freq*patch_shift_freq
				
				patch = np.copy(norm_data[p_freq:p_freq+freq_size, p_dec:p_dec+sky_size, p_ra:p_ra+sky_size])

				input_test[i,0:freq_size*sky_size*sky_size] = (patch.flatten("C")/65535.0)*2.0 - 1.0
				
				cat_id = np.where((coords[:,0] >= p_ra)   & (coords[:,1] <= p_ra+sky_size) &\
								  (coords[:,2] >= p_dec)  & (coords[:,3] <= p_dec+sky_size) &\
								  (coords[:,4] >= p_freq) & (coords[:,5] <= p_freq+freq_size))[0]
				
				targets_test[i,:] = 0.0
				targets_test[i,0] = min(max_nb_obj_per_image, np.shape(cat_id)[0])
				if(np.shape(cat_id)[0] >= max_nb_obj_per_image):
					print ("Max obj. reached")
				
				for k in range(0, int(targets_test[i,0])):
					l_t_pa = np.copy(pa_list[cat_id[k]])
				
					l_coords = np.copy(coords[cat_id[k],:])
					l_coords[0:2] -= p_ra
					l_coords[2:4] -= p_dec
					l_coords[4:6] -= p_freq
					
					targets_test[i,1+k*(7+nb_param+diff_flagging)] = 1.0 #target class
					targets_test[i,1+k*(7+nb_param+diff_flagging)+1:1+k*(7+nb_param+diff_flagging)+7] = l_coords[[0,2,4,1,3,5]] #reordered boxes coords
					targets_test[i,1+k*(7+nb_param+diff_flagging)+7:1+k*(7+nb_param+diff_flagging)+7+nb_param] = \
						np.array([flux_list[cat_id[k]],\
								  hi_size_list[cat_id[k]],\
								  w20_list[cat_id[k]],
								  (np.sin(l_t_pa*np.pi/180.0)+1)*0.5,\
								  (np.cos(l_t_pa*np.pi/180.0)+1)*0.5,\
								  np.cos(inc_list[cat_id[k]]*np.pi/180.0)])
					if(diff_flagging and bootstrap):
						targets_test[i,1+k*(7+nb_param+diff_flagging)+7+nb_param] = diff_list[cat_id[k]]
	
	return input_test, targets_test
	

def process_pred(process_file, save_file):

	pred_data = np.fromfile(process_file, dtype="float32")
	pred_data = np.reshape(pred_data, (nb_area_freq, nb_area_sky, nb_area_sky, nb_box*(8+nb_param), yolo_nb_freq_reg, yolo_nb_sky_reg, yolo_nb_sky_reg))

	final_boxes = []

	c_tile = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")
	c_tile_kept = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")
	c_box = np.zeros((8+1+nb_param+1),dtype="float32")

	start_time = time.time()

	for p_freq in range(0,nb_area_freq):
		for p_dec in range(0,nb_area_sky):
			for p_ra in range(0,nb_area_sky):
				
				c_tile[:,:] = 0.0
				c_tile_kept[:,:] = 0.0
				c_pred = pred_data[p_freq,p_dec,p_ra,:,:,:]
				
				c_nb_box = tile_filter(c_pred, c_box, c_tile, nb_box, nb_param, yolo_nb_sky_reg, yolo_nb_freq_reg)
				
				c_nb_box_final = c_nb_box
				c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, 0.1)
				
				final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))
				
	final_boxes = np.reshape(np.array(final_boxes, dtype="object"), (nb_area_freq, nb_area_sky, nb_area_sky))

	c_tile = np.zeros((yolo_nb_sky_reg*yolo_nb_sky_reg*yolo_nb_freq_reg*nb_box,(8+1+nb_param+1)),dtype="float32")

	A = np.array([-1, 0, 1])
	B = np.array([-1, 0, 1])
	C = np.array([-1, 0, 1])

	x, y, z = np.meshgrid(A, B, C)
	dir_array = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
	l_overlap = np.array((overlap_sky, overlap_sky, overlap_freq))
	l_patch_shift = np.array((patch_shift_sky, patch_shift_sky, patch_shift_freq))
	l_patch_size = np.array((sky_size, sky_size, freq_size))

	start_time = time.time()

	#Second NMS over all the overlapping patches
	for p_freq in range(0,nb_area_freq):
		for p_dec in range(0,nb_area_sky):
			for p_ra in range(0,nb_area_sky):
				boxes = np.copy(final_boxes[p_freq,p_dec,p_ra])
				for l in range(0,np.shape(dir_array)[0]):
					if(p_freq+dir_array[l,2] >= 0 and p_freq+dir_array[l,2] <= nb_area_freq-1 and\
					   p_dec +dir_array[l,1] >= 0 and p_dec +dir_array[l,1] <= nb_area_sky-1  and\
					   p_ra  +dir_array[l,0] >= 0 and p_ra  +dir_array[l,0] <= nb_area_sky-1 ):
						comp_boxes = np.copy(final_boxes[p_freq+dir_array[l,2],p_dec+dir_array[l,1],p_ra+dir_array[l,0]])
						c_nb_box = inter_patch_NMS(boxes, comp_boxes, c_tile, dir_array[l], l_overlap, l_patch_shift, l_patch_size, -0.3)
						boxes = np.copy(c_tile[0:c_nb_box,:])
				
				final_boxes[p_freq,p_dec,p_ra] = np.copy(boxes)

	# Convert back to full image pixel coordinates
	final_boxes_scaled = np.copy(final_boxes)
	for p_freq in range(0,nb_area_freq):
		box_freq_offset = p_freq*patch_shift_freq
		for p_dec in range(0,nb_area_sky):
			box_dec_offset = p_dec*patch_shift_sky
			for p_ra in range(0,nb_area_sky):
				box_ra_offset = p_ra*patch_shift_sky
				
				final_boxes_scaled[p_freq,p_dec,p_ra][:,0] = box_ra_offset   + final_boxes_scaled[p_freq,p_dec,p_ra][:,0] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,1] = box_dec_offset  + final_boxes_scaled[p_freq,p_dec,p_ra][:,1] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,2] = box_freq_offset + final_boxes_scaled[p_freq,p_dec,p_ra][:,2] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,3] = box_ra_offset   + final_boxes_scaled[p_freq,p_dec,p_ra][:,3] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,4] = box_dec_offset  + final_boxes_scaled[p_freq,p_dec,p_ra][:,4] - 0.5
				final_boxes_scaled[p_freq,p_dec,p_ra][:,5] = box_freq_offset + final_boxes_scaled[p_freq,p_dec,p_ra][:,5] - 0.5

	flat_kept_scaled = np.vstack(final_boxes_scaled.flatten())
	flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:,7].argsort(),:][::-1]

	np.savetxt(save_file, flat_kept_scaled)
	

def assemble_and_build_catalog():
	
	box_cat = np.loadtxt("filtered_pred_ldev.txt")
	
	# Remove orig offset
	box_cat[:,0] -= orig_offset_sky
	box_cat[:,1] -= orig_offset_sky
	box_cat[:,2] -= orig_offset_freq
	box_cat[:,3] -= orig_offset_sky
	box_cat[:,4] -= orig_offset_sky
	box_cat[:,5] -= orig_offset_freq
	
	np.savetxt("net_pred_filtered_repos_rescaled_ldev.dat", box_cat)
	
	hdul = fits.open("/home/trzguo/content/SDC2/Cornu_et_al_2026_SDC2_models_catalogs_and_codes_archive/codes/complete_training_and_inference_pipeline/data/merged_cube.fits", memmap=True)
	wcs_cube = WCS(hdul[0].header)
	
	cls = utils.pixel_to_skycoord((box_cat[:,3]+box_cat[:,0])*0.5, (box_cat[:,4]+box_cat[:,1])*0.5, wcs_cube)
	ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])

	cat_size = int(np.shape(box_cat)[0])

	cat_header = "id ra dec hi_size line_flux_integral central_freq pa i w20"
	final_box_cat = np.zeros((cat_size,9), dtype="float32")

	lims = np.loadtxt("train_cat_lims.txt")

	final_box_cat[:,0] = np.arange(0,cat_size)
	final_box_cat[:,[1,2]] = ra_dec_coords.T
	final_box_cat[:,3] = box_cat[:,10]*lims[1,0] + lims[1,1]
	final_box_cat[:,4] = np.exp(box_cat[:,9]*lims[0,0] + lims[0,1])
	final_box_cat[:,5] = (box_cat[:,5]+box_cat[:,2])*0.5*pixel_size_freq + 9.5e8
	final_box_cat[:,6] = np.mod(np.arctan2(np.clip(box_cat[:,12],0.0,1.0)*2.0-1.0, np.clip(box_cat[:,13],0.0,1.0)*2.0-1.0)*180.0/np.pi,360.0)
	final_box_cat[:,7] = np.arccos(np.clip(box_cat[:,14],0.0,1.0))*180.0/np.pi
	final_box_cat[:,8] = (np.exp(box_cat[:,11]*lims[2,0] + lims[2,1])*pixel_size_freq)/(final_box_cat[:,5]**2/1.4204e9)*299792.458
	
	np.savetxt("alfalfa_catalog.txt", final_box_cat, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f")

	cat_header = "id ra dec hi_size line_flux_integral central_freq pa i w20"


def scoring():
	
	# Score over the training area to build the matching catalog used in bootstrap. 
	# For scoring over the full datacube, refer to full_data_gen.py
	
	cat_header = "id ra dec hi_size line_flux_integral central_freq pa i w20"

	pred_cat = np.loadtxt("alfalfa_catalog.txt", skiprows=1)
	true_cat = np.loadtxt("alfalfa_full_catalog.txt", skiprows=1)
	raw_cat = np.loadtxt("net_pred_filtered_repos_rescaled_ldev.dat")
	
	min_obj = 0.6
	max_size = np.shape(pred_cat)[0] - np.searchsorted(raw_cat[:,7][::-1], min_obj)
	pred_cat = pred_cat[:max_size]
	raw_cat = raw_cat[:max_size]
	per_source_score = np.zeros((max_size)) - 1.0
	
	np.savetxt("pre_opt_cat.txt", pred_cat, header=cat_header, comments="", fmt="%d %3.13f %2.13f %1.13f %1.13f %10.1f %3.13f %2.13f %3.13f")
	
	sub_cat_path = "pre_opt_cat.txt"
	truth_cat_path = "sky_ldev_truthcat_v2.txt"

	scorer = Sdc2Scorer.from_txt(sub_cat_path, truth_cat_path, sub_skiprows=0, truth_skiprows=0)
	scorer.run(detail=True)
	
	score_details = scorer.score.scores_df
	
	print("Final score: {}".format(scorer.score.value))
	
	print ("Ndet:", scorer.score.n_det)
	print ("Nmatch:", scorer.score.n_match)
	print ("Nbad:", scorer.score.n_bad)
	print ("Nfalse:", scorer.score.n_false)
	
	print ("Score det:",scorer.score.score_det)
	print ("Accuracy:",scorer.score.acc_pc)
	print ("Purity:", (scorer.score.n_match/(scorer.score.n_match+scorer.score.n_false)))
	
	score_details = scorer.score.scores_df
	
	print (np.mean(score_details["position"]))
	print (np.mean(score_details["central_freq"]))
	print (np.mean(score_details["flux"]))
	print (np.mean(score_details["hi_size"]))
	print (np.mean(score_details["pa"]))
	print (np.mean(score_details["w20"]))
	print (np.mean(score_details["i"]))
	
	#Identify matching sources for bootstrap
	matches = scorer.score.match_df
	np.savetxt("matches_id_targ.txt", matches["id_t"])
	










