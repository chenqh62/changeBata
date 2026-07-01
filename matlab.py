import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
# 导入配置文件常量，解决NameError
from record_config import MIRROR_ITER_ORDER, THEORY_MIRROR_ORDER
# 中文显示设置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# 读取仿真CSV数据
df = pd.read_csv("带电小球运动数据.csv")
t = df["时间(s)"]
vA = df["A_v(m/s)"]
vB = df["B_v(m/s)"]
F_sim = df["模拟阶数N库仑力(N)"]
Ep_sim = df["模拟N阶电势能Ep_sim(J)"]
Ep_theo = df["理论高阶Ep_theory(J)"]
E_sim = df["模拟总机械能E_sim(J)"]
E_theo = df["理论总机械能E_theory(J)"]
dEp_pct = df["Ep相对误差δEp(%)"]
dE_pct = df["E总相对误差δE(%)"]

# ========== 新增平滑插值 ==========
t_smooth = np.linspace(t.min(), t.max(), 300)
# 速度插值
spl_A = make_interp_spline(t, vA, k=3)
vA_smooth = spl_A(t_smooth)
spl_B = make_interp_spline(t, vB, k=3)
vB_smooth = spl_B(t_smooth)
# 库仑力插值
spl_F = make_interp_spline(t, F_sim, k=3)
F_smooth = spl_F(t_smooth)
# 势能插值
spl_EpS = make_interp_spline(t, Ep_sim, k=3)
EpS_smooth = spl_EpS(t_smooth)
spl_EpT = make_interp_spline(t, Ep_theo, k=3)
EpT_smooth = spl_EpT(t_smooth)
# 总机械能插值
spl_ES = make_interp_spline(t, E_sim, k=3)
ES_smooth = spl_ES(t_smooth)
spl_ET = make_interp_spline(t, E_theo, k=3)
ET_smooth = spl_ET(t_smooth)
# 误差插值
spl_dE = make_interp_spline(t, dE_pct, k=3)
dE_smooth = spl_dE(t_smooth)
# ==================================

# 碰撞时刻筛选
collide_mask = df["是否碰撞"] == "Y"
t_first = df.loc[collide_mask, "时间(s)"].iloc[0] if len(df.loc[collide_mask])>0 else None
# 误差统计输出
print("==================== 迭代镜像法截断误差评估报告 ====================")
print(f"当前仿真镜像阶数 N = {MIRROR_ITER_ORDER}")
print(f"理论真值参考阶数 = {THEORY_MIRROR_ORDER}")
print("【电势能相对误差】")
print(f"最大误差：{dEp_pct.max():.4f} %")
print(f"平均误差：{dEp_pct.mean():.4f} %")
print("【系统总机械能相对误差（核心评估指标）】")
print(f"最大误差：{dE_pct.max():.4f} %")
print(f"平均误差：{dE_pct.mean():.4f} %")
print("====================================================================")
# 画布布局 5张子图
plt.figure(figsize=(14, 14))
# 子图1：速度-时间（平滑曲线）
plt.subplot(5,1,1)
plt.plot(t_smooth, vA_smooth, c="#e63946", lw=1.5, label="小球A速度")
plt.plot(t_smooth, vB_smooth, c="#457b9d", lw=1.5, label="小球B速度")
if t_first:
    plt.axvline(t_first, c="k", ls="--", label="首次碰撞")
plt.title("两小球速度-时间曲线（平滑插值）")
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.grid(alpha=0.3)
plt.legend()
# 子图2：镜像库仑力
plt.subplot(5,1,2)
plt.plot(t_smooth, F_smooth, c="#f77f00", lw=1.5, label=f"{MIRROR_ITER_ORDER}阶镜像库仑力")
if t_first:
    plt.axvline(t_first, c="k", ls="--")
plt.title("迭代镜像法库仑力变化")
plt.xlabel("t (s)")
plt.ylabel("F (N)")
plt.grid(alpha=0.3)
plt.legend()
# 子图3：电势能对比
plt.subplot(5,1,3)
plt.plot(t_smooth, EpS_smooth, c="#2b9348", lw=1.5, label=f"{MIRROR_ITER_ORDER}阶模拟势能")
plt.plot(t_smooth, EpT_smooth, c="#c1121f", lw=1.5, label=f"{THEORY_MIRROR_ORDER}阶理论真值势能")
if t_first:
    plt.axvline(t_first, c="red", ls="--")
plt.title("电势能：模拟值 VS 理论精确解")
plt.xlabel("t (s)")
plt.ylabel("Ep (J)")
plt.grid(alpha=0.3)
plt.legend()
# 子图4：总机械能对比
plt.subplot(5,1,4)
plt.plot(t_smooth, ES_smooth, c="#2b9348", lw=1.5, label="有限阶镜像模拟总机械能")
plt.plot(t_smooth, ET_smooth, c="#c1121f", lw=1.5, label="无穷镜像近似理论总机械能")
if t_first:
    plt.axvline(t_first, c="k", ls="--")
plt.title("系统总机械能对比（模拟值vs理论真值）")
plt.xlabel("t (s)")
plt.ylabel("E (J)")
plt.grid(alpha=0.3)
plt.legend()
# 子图5：误差曲线（核心）
plt.subplot(5,1,5)
plt.plot(t_smooth, dE_smooth, c="#2b9348", lw=1.5, label=f"N={MIRROR_ITER_ORDER}镜像法相对误差")
plt.axhline(dE_pct.mean(), c="orange", ls="-.", label=f"平均误差={dE_pct.mean():.4f}%")
plt.axhline(dE_pct.max(), c="crimson", ls="-.", label=f"最大误差={dE_pct.max():.4f}%")
if t_first:
    plt.axvline(t_first, c="k", ls="--")
plt.title("迭代镜像法自身截断相对误差曲线（评估方法可行性）")
plt.xlabel("t (s)")
plt.ylabel("相对误差 δ (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()