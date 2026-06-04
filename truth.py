import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u

# ===================== 配置 =====================
TXT_PATH = "apjaac956t2_mrt.txt"   # 你的ALFALFA原始表
SAVE_PATH = "alfalfa_full_catalog.txt"
MATCH_RADIUS = 10 * u.arcsec       # 匹配半径 10角秒（此参数已无实际作用，可保留）
Vizier.ROW_LIMIT = -1              # 不限制行数（此参数已无实际作用，可保留）
# ================================================

# ---------- 工具函数：时分秒/度分秒 转十进制度 ----------
def hms2deg(h, m, s):
    return h + m/60.0 + s/3600.0
def dms2deg(sign, d, m, s):
    dec = d + m/60.0 + s/3600.0
    return dec.where(sign != "-", -dec)  # 用 where 批量处理符号

# ---------- 1. 读取ALFALFA固定宽度数据表 ----------
colspecs = [
    (0, 6),     # AGCNr
    (16, 18),   # HIRAh
    (19, 21),   # HIRAm
    (22, 26),   # HIRAs
    (27, 28),   # HIDE-
    (28, 30),   # HIDEd
    (31, 33),   # HIDEm
    (34, 36),   # HIDEs
    (58, 63),   # Vhelio
    (64, 67),   # W50
    (72, 75),   # W20
    (76, 82),   # HIflux
]
names = [
    "AGCNr","HIRAh","HIRAm","HIRAs",
    "HIDE_sign","HIDEd","HIDEm","HIDEs",
    "Vhelio","W50","W20","HIflux"
]

df_alfa = pd.read_fwf(
    TXT_PATH,
    colspecs=colspecs,
    names=names,
    skiprows=0,
    na_values=["", " "]
)

# 计算 J2000 赤经/赤纬 十进制度
df_alfa["ra"] = hms2deg(df_alfa["HIRAh"], df_alfa["HIRAm"], df_alfa["HIRAs"])
df_alfa["dec"] = dms2deg(df_alfa["HIDE_sign"], df_alfa["HIDEd"], df_alfa["HIDEm"], df_alfa["HIDEs"])

before_filter = len(df_alfa)
df_alfa = df_alfa[df_alfa["dec"] >= 31.0]  # 这里就是过滤
after_filter = len(df_alfa)
print(f"✅ 过滤完成：保留 Dec≥31° 的源 {after_filter} 个")
print(f"❌ 剔除 Dec<31° 的源 {before_filter - after_filter} 个")
# ==============================================


# 重命名为你需要的字段
df_alfa = df_alfa.rename(columns={
    "AGCNr": "id",
    "Vhelio": "central_freq",
    "W50": "hi_size",
    "HIflux": "line_flux_integral",
    "W20": "w20"
})
# 仅保留核心字段
df_alfa = df_alfa[["id","ra","dec","hi_size","line_flux_integral","central_freq","w20"]].dropna(subset=["ra","dec"])

# ---------- 2. 直接设置固定的pa和倾角(i) ----------
# 新增pa字段，所有值设为180
df_alfa["pa"] = 180
# 新增inclination字段（对应输出的i），所有值设为45
df_alfa["i"] = 45

df_alfa["id"] = range(len(df_alfa))

# ---------- 4. 保存最终完整表 ----------
# 修正out_cols：将"i"改为"inclination"（原代码的bug）
out_cols = ["id","ra","dec","hi_size","line_flux_integral","central_freq","pa","i","w20"]
df_final = df_alfa[out_cols].copy()
# 可选：如果需要将inclination重命名为i，添加这行
# df_final = df_final.rename(columns={"inclination": "i"})
df_final.to_csv(
    "alfalfa_full_catalog.txt",
    sep=" ",
    index=False,
    header=True  # 绝对不能有表头！
)

print(f"匹配完成，有效源数量：{len(df_final)}")
print(f"结果已保存至：{SAVE_PATH}")
print("\n前5行预览：")
print(df_final.head())