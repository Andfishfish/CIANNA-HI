import numpy as np
from astropy.io import fits
import glob
import os
import re

# ====================== 【只需改这里】 ======================
input_dir = "/home/yjjiao/data/CRAFTS/cube/Dec+3135_01_05/20220330/"          # 所有fits文件所在文件夹
output_dir = "/home/trzguo/content/SDC2/Cornu_et_al_2026_SDC2_models_catalogs_and_codes_archive/codes/complete_training_and_inference_pipeline/data/Dec+3135_01_05"  # 输出文件夹（自动创建）
CDELT3 = 0.00762939453125 # 频率步长（你的数据固定值）
# ============================================================

# ----------------------
# 1. 读取所有文件并自动提取组号（0001 ~ 004X）
# ----------------------
all_files = glob.glob(os.path.join(input_dir, "*.fits"))

# 正则提取 组号（如 0001、0002...）
def get_group_id(filename):
    match = re.search(r"_(\d{4})_", filename)
    if match:
        return match.group(1)
    return None

group_files = {}
for f in all_files:
    gid = get_group_id(f)
    if gid:
        if gid not in group_files:
            group_files[gid] = []
        group_files[gid].append(f)

# 打印找到的组数
print(f"✅ 总共找到 {len(group_files)} 组数据需要处理")
print(f"组号列表: {sorted(group_files.keys())}\n")

# ----------------------
# 2. 对每一组数据：自动匹配 pz13 / pz23
# ----------------------
for gid in sorted(group_files.keys()):
    print(f"==================================================")
    print(f"🚀 开始处理第 {gid} 组数据")
    
    files = group_files[gid]
    pz13 = sorted([f for f in files if "pz13" in f])
    pz23 = sorted([f for f in files if "pz23" in f])

    if not (len(pz13) == len(pz23)):
        print(f"❌ 组 {gid} 文件不匹配，跳过")
        continue

    print(f"✅ 本组找到 {len(pz13)} 个频率块")

    # ----------------------
    # 3. 每组内：计算 Stokes I = (pz13 + pz23)/2 * mask
    # ----------------------
    cube_list = []
    header_list = []
    
    for i in range(len(pz13)):
        with fits.open(pz13[i]) as hdul:
            d13 = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header.copy()
        with fits.open(pz23[i]) as hdul:
            d23 = hdul[0].data.astype(np.float32)
        
        both_valid = ~np.isnan(d13) & ~np.isnan(d23)
        stokesI = np.where(both_valid, 
                       (d13 + d23) / 2.0,  # 都有效 → 平均
                       np.nan_to_num(d13, nan=0.0) + np.nan_to_num(d23, nan=0.0))  # 一个有效 → 取非NaN
        cube_list.append(stokesI)
        header_list.append(hdr)

    # ----------------------
    # 4. 频率轴拼接（自动无缝合并）
    # ----------------------
    freq_starts = [h["CRVAL3"] for h in header_list]
    naxis3_list = [h["NAXIS3"] for h in header_list]
    freq_ends = [s + (n-1)*CDELT3 for s,n in zip(freq_starts, naxis3_list)]
    
    freq_min = min(freq_starts)
    freq_max = max(freq_ends)
    naxis3_new = int(np.round((freq_max - freq_min)/CDELT3)) + 1
    _, ny, nx = cube_list[0].shape

    merged_cube = np.zeros((naxis3_new, ny, nx), dtype=np.float32)
    count_cube = np.zeros((naxis3_new, ny, nx), dtype=np.int16)

    for i, cube in enumerate(cube_list):
        s = header_list[i]["CRVAL3"]
        n = header_list[i]["NAXIS3"]
        idx_start = int(np.round((s - freq_min) / CDELT3))
        merged_cube[idx_start:idx_start+n] += cube
        count_cube[idx_start:idx_start+n] += 1

    count_cube[count_cube == 0] = 1
    merged_cube /= count_cube

    # ----------------------
    # 5. 保存最终 FITS cube
    # ----------------------
    outfile = os.path.join(output_dir, f"final_cube_{gid}.fits")
    
    final_hdr = header_list[0].copy()
    final_hdr["NAXIS3"] = naxis3_new
    final_hdr["CRVAL3"] = freq_min
    final_hdr["CRPIX3"] = 1.0
    final_hdr["CDELT3"] = CDELT3
    final_hdr["BUNIT"] = "Jy/beam"
    final_hdr["COMMENT"] = "Stokes I = (pz13+pz23)/2 + freq merged"

    fits.PrimaryHDU(merged_cube, final_hdr).writeto(outfile, overwrite=True)

    print(f"✅ 组 {gid} 处理完成！")
    print(f"📦 输出: {outfile}")
    print(f"📊 形状 [freq, DEC, RA] = [{naxis3_new}, {ny}, {nx}]\n")

print("==================================================")
print("🎉 🎯 所有 40+ 组数据全部处理完成！")
print(f"💾 输出文件在: {output_dir}")
print("👉 每个文件都是可直接用于训练的最终数据立方体！")