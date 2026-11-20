import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# 定义 Hernquist 模型
def fHern(r, rhoH, rH):
    return rhoH / ( (r/rH) * (1 + r/rH)**3 )

# 初始化结果存储
results = {
    'snap': [],
    'rhoH_fit': [],
    'rH_fit': [],
    'rhoH_err': [],
    'rH_err': []
}

# 设定半径范围
r_min, r_max = 0.5, 50.0

# 循环读取并拟合每个 snapshot
for i in range(20):  # snap000 到 snap035
    snap_id = f"snap{str(i).zfill(3)}"  # 补零，如 snap000
    file_name = f"prof_bound_stars_snapshot_{snap_id}.txt"
    
    try:
        # 读取数据
        data = np.genfromtxt(file_name, names=True)
        
        # 筛选半径范围
        mask = (data['r'] >= r_min) & (data['r'] <= r_max)
        r_data = data['r'][mask]
        rho_data = data['rho'][mask]
        
        # 初始猜测（基于当前数据动态计算）
        peak_idx = np.argmax(rho_data)
        rH_guess = r_data[peak_idx]  # 特征半径≈密度峰值位置
        rhoH_guess = rho_data[peak_idx] * (r_data[peak_idx]/rH_guess) * (1 + r_data[peak_idx]/rH_guess)**3
        
        # 执行拟合
        params, cov = curve_fit(
            fHern,
            r_data,
            rho_data,
            p0=[rhoH_guess, rH_guess],
            bounds=[(0, 0), (np.inf, np.inf)]
        )
        
        # 存储结果
        results['snap'].append(i)
        results['rhoH_fit'].append(params[0])
        results['rH_fit'].append(params[1])
        results['rhoH_err'].append(np.sqrt(cov[0, 0]))
        results['rH_err'].append(np.sqrt(cov[1, 1]))
        
        print(f"{snap_id}: ρH = {params[0]:.2e}, rH = {params[1]:.2f}")
    
    except FileNotFoundError:
        print(f"Warning: {file_name} not found. Skipping.")
        continue

# 转为 DataFrame 并保存为 CSV
df_results = pd.DataFrame(results)
df_results.to_csv("hernquist_fit_results.csv", index=False)
print("Results saved to 'hernquist_fit_results.csv'")

