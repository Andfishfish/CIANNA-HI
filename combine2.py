import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import glob
import os

def read_fits_info(file_list):
    all_data = []
    all_headers = []
    for file in file_list:
        with fits.open(file, memmap=True) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            if len(data.shape) != 3:
                raise ValueError(f"文件 {file} 不是3D数据（shape={data.shape}）")
            all_data.append(data)
            all_headers.append(header)
    return all_data, all_headers

def check_consistency(all_headers, file_list):
    ref_header = all_headers[0]
    ref_cdelt = [ref_header[f'CDELT{i+1}'] for i in range(3)]
    ref_ctype = [ref_header[f'CTYPE{i+1}'] for i in range(3)]
    ref_file = file_list[0]

    for idx, (header, file) in enumerate(zip(all_headers, file_list)):
        current_cdelt = [header[f'CDELT{i+1}'] for i in range(3)]
        current_ctype = [header[f'CTYPE{i+1}'] for i in range(3)]

        if not np.allclose(current_cdelt, ref_cdelt):
            raise ValueError(
                f"像素分辨率（CDELT）不一致！参考文件: {ref_file}\n"
                f"异常文件: {file}\n"
                f"参考CDELT: {ref_cdelt}\n"
                f"异常CDELT: {current_cdelt}"
            )

        if not np.array_equal(current_ctype, ref_ctype):
            raise ValueError(
                f"坐标类型（CTYPE）不一致！参考文件: {ref_file}\n"
                f"异常文件: {file}\n"
                f"参考CTYPE: {ref_ctype}\n"
                f"异常CTYPE: {current_ctype}"
            )

    print("✅ 所有文件 CDELT/CTYPE 检查通过")
    return ref_cdelt

def get_axis_ranges(all_headers, all_data):
    print("\n--- 计算全局坐标范围 ---")
    axis_ranges = []
    for axis in range(3):
        all_min = []
        all_max = []
        for header, data in zip(all_headers, all_data):
            naxis = data.shape[::-1][axis]
            crval = header[f'CRVAL{axis+1}']
            cdelt = header[f'CDELT{axis+1}']
            crpix = header[f'CRPIX{axis+1}']
            # 手动计算该文件的物理坐标范围
            pix_min = 0
            pix_max = naxis - 1
            world_min = crval + (pix_min - crpix + 1) * cdelt
            world_max = crval + (pix_max - crpix + 1) * cdelt
            all_min.append(min(world_min, world_max))
            all_max.append(max(world_min, world_max))
        axis_ranges.append((np.min(all_min), np.max(all_max)))
        print(f"轴{axis+1} 范围: [{np.min(all_min):.4f}, {np.max(all_max):.4f}]")
    print("-------------------------\n")
    return axis_ranges

def compute_new_header(axis_ranges, cdelt, ref_header):
    new_header = fits.Header()
    new_header['SIMPLE'] = True
    new_header['BITPIX'] = -32
    new_header['NAXIS'] = 3
    new_naxis = []
    new_crval = []
    new_crpix = []
    for i in range(3):
        min_val, max_val = axis_ranges[i]
        cd = cdelt[i]
        # 计算新的像素数
        npix = int(np.round((max_val - min_val) / abs(cd))) + 1
        new_naxis.append(npix)
        # 设置新的CRVAL（取范围的一端，根据CDELT方向）
        if cd > 0:
            start_val = min_val
        else:
            start_val = max_val
        new_crval.append(start_val)
        new_crpix.append(1.0)

        new_header[f'NAXIS{i+1}'] = npix
        new_header[f'CTYPE{i+1}'] = ref_header[f'CTYPE{i+1}']
        new_header[f'CRVAL{i+1}'] = start_val
        new_header[f'CRPIX{i+1}'] = 1.0
        new_header[f'CDELT{i+1}'] = cd
        if f'CUNIT{i+1}' in ref_header:
            new_header[f'CUNIT{i+1}'] = ref_header[f'CUNIT{i+1}']

    # 复制关键元数据
    for key in ['BUNIT', 'RESTFRQ', 'EQUINOX', 'RADESYS']:
        if key in ref_header:
            new_header[key] = ref_header[key]

    print("\n--- 新WCS信息 ---")
    print(f"NAXIS: {new_naxis}")
    print(f"CRVAL: {new_crval}")
    print(f"CDELT: {cdelt}")
    print("----------------\n")
    return new_header, new_naxis

def merge_fits(file_list, output_file):
    print("正在读取文件信息...")
    all_data, all_headers = read_fits_info(file_list)

    print("正在检查核心参数一致性...")
    cdelt = check_consistency(all_headers, file_list)

    print("正在计算拼接总范围...")
    axis_ranges = get_axis_ranges(all_headers, all_data)

    print("正在生成全局坐标信息...")
    new_header, new_naxis = compute_new_header(axis_ranges, cdelt, all_headers[0])
    new_shape = tuple(new_naxis[::-1])
    print(f"✅ 拼接后数组形状: {new_shape}")

    sum_data = np.zeros(new_shape, dtype=np.float32)
    count_data = np.zeros(new_shape, dtype=np.int32)

    print("开始拼接数据...")
    for idx, (data, header, file) in enumerate(zip(all_data, all_headers, file_list), 1):
        print(f"\n处理 {idx}/{len(all_data)}: {os.path.basename(file)} | 原始形状: {data.shape}")
        
        nfreq, ndec, nra = data.shape
        print(f"  数据维度：Freq={nfreq}, Dec={ndec}, RA={nra}")

        # 1. 生成每个文件的物理坐标（RA, Dec, Freq）
        # RA轴
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header['CRPIX1']
        pix_ra = np.arange(nra)
        world_ra = crval1 + (pix_ra - crpix1 + 1) * cdelt1

        # Dec轴
        crval2 = header['CRVAL2']
        cdelt2 = header['CDELT2']
        crpix2 = header['CRPIX2']
        pix_dec = np.arange(ndec)
        world_dec = crval2 + (pix_dec - crpix2 + 1) * cdelt2

        # Freq轴
        crval3 = header['CRVAL3']
        cdelt3 = header['CDELT3']
        crpix3 = header['CRPIX3']
        pix_freq = np.arange(nfreq)
        world_freq = crval3 + (pix_freq - crpix3 + 1) * cdelt3

        print(f"  原始文件坐标范围：")
        print(f"    RA: [{np.min(world_ra):.4f}, {np.max(world_ra):.4f}]")
        print(f"    DEC: [{np.min(world_dec):.4f}, {np.max(world_dec):.4f}]")
        print(f"    FREQ: [{np.min(world_freq):.4f}, {np.max(world_freq):.4f}]")

        # 2. 手动计算在全局数组中的像素坐标（绕开world_to_pixel）
        # 全局WCS参数
        new_crval1 = new_header['CRVAL1']
        new_cdelt1 = new_header['CDELT1']
        new_crpix1 = new_header['CRPIX1']

        new_crval2 = new_header['CRVAL2']
        new_cdelt2 = new_header['CDELT2']
        new_crpix2 = new_header['CRPIX2']

        new_crval3 = new_header['CRVAL3']
        new_cdelt3 = new_header['CDELT3']
        new_crpix3 = new_header['CRPIX3']

        # 生成网格
        world_ra_grid, world_dec_grid, world_freq_grid = np.meshgrid(world_ra, world_dec, world_freq, indexing='ij')
        world_ra_flat = world_ra_grid.flatten()
        world_dec_flat = world_dec_grid.flatten()
        world_freq_flat = world_freq_grid.flatten()

        # 计算全局像素坐标
        new_pix1 = (world_ra_flat - new_crval1) / new_cdelt1 + new_crpix1 - 1
        new_pix2 = (world_dec_flat - new_crval2) / new_cdelt2 + new_crpix2 - 1
        new_pix3 = (world_freq_flat - new_crval3) / new_cdelt3 + new_crpix3 - 1

        print(f"  转换后像素范围：")
        print(f"    pix1 (RA): [{np.min(new_pix1):.4f}, {np.max(new_pix1):.4f}]")
        print(f"    pix2 (DEC): [{np.min(new_pix2):.4f}, {np.max(new_pix2):.4f}]")
        print(f"    pix3 (FREQ): [{np.min(new_pix3):.4f}, {np.max(new_pix3):.4f}]")

        # 取整
        new_pix1 = np.round(new_pix1).astype(int)
        new_pix2 = np.round(new_pix2).astype(int)
        new_pix3 = np.round(new_pix3).astype(int)

        # 过滤有效像素
        valid_mask = (
            (new_pix1 >= 0) & (new_pix1 < new_naxis[0]) &
            (new_pix2 >= 0) & (new_pix2 < new_naxis[1]) &
            (new_pix3 >= 0) & (new_pix3 < new_naxis[2])
        )

        valid_count = np.sum(valid_mask)
        print(f"  有效像素数: {valid_count} / {len(valid_mask)}")
        if valid_count == 0:
            print("  ❌ 无有效像素！")
            continue

        print(f"  有效像素占比: {valid_count/len(valid_mask)*100:.2f}%")

        # 处理数据
        data_flat = data.flatten()
        valid_data = data_flat[valid_mask]
        non_nan_mask = ~np.isnan(valid_data)
        print(f"  非NaN像素数: {np.sum(non_nan_mask)} / {len(valid_data)}")

        if np.any(non_nan_mask):
            idx1 = new_pix1[valid_mask][non_nan_mask]
            idx2 = new_pix2[valid_mask][non_nan_mask]
            idx3 = new_pix3[valid_mask][non_nan_mask]
            sum_data[idx3, idx2, idx1] += valid_data[non_nan_mask]
            count_data[idx3, idx2, idx1] += 1

    print("\n正在计算最终拼接结果...")
    result_data = np.full(new_shape, np.nan, dtype=np.float32)
    count_mask = count_data > 0
    result_data[count_mask] = sum_data[count_mask] / count_data[count_mask]

    nan_fraction = np.isnan(result_data).sum() / result_data.size
    print(f"拼接完成！结果NaN占比: {nan_fraction*100:.2f}%")

    hdu = fits.PrimaryHDU(result_data, header=new_header)
    hdu.writeto(output_file, overwrite=True)
    print(f"\n🎉 文件保存至: {output_file}")

if __name__ == "__main__":
    # 自动匹配当前目录所有fits文件，可手动修改为 file_list = ["1.fits", "2.fits"]
    file_list = glob.glob("/home/trzguo/content/SDC2/Cornu_et_al_2026_SDC2_models_catalogs_and_codes_archive/codes/complete_training_and_inference_pipeline/data/**/*.fits", recursive=True)
    if not file_list:
        print("❌ 错误：未找到FITS文件")
        exit(1)

    print(f"找到 {len(file_list)} 个待拼接FITS文件")
    merge_fits(file_list, output_file="/home/trzguo/content/SDC2/Cornu_et_al_2026_SDC2_models_catalogs_and_codes_archive/codes/complete_training_and_inference_pipeline/data/merged_cube.fits")