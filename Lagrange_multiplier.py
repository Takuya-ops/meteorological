import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import japanize_matplotlib

# 観測データ
temp_obs = 25.0  # 温度の観測値 (°C)
humidity_obs = 60.0  # 湿度の観測値 (%)


# 物理的な制約（簡略化した例）
def constraint(x):
    temp, humidity = x
    return temp + humidity - 100  # 温度と湿度の和が100以下という仮の制約


# 目的関数（観測値との差の二乗和を最小化）
def objective(x):
    temp, humidity = x
    return (temp - temp_obs) ** 2 + (humidity - humidity_obs) ** 2


# 初期推測
x0 = [temp_obs, humidity_obs]

# 制約条件
con = {"type": "ineq", "fun": lambda x: -constraint(x)}

# 最適化
res = minimize(objective, x0, method="SLSQP", constraints=con)

# グラフの作成
plt.figure(figsize=(10, 8))

# 制約条件のプロット
temp_range = np.linspace(0, 100, 100)
humidity_range = 100 - temp_range
plt.plot(temp_range, humidity_range, "r--", label="制約条件")

# 観測点のプロット
plt.scatter(temp_obs, humidity_obs, color="blue", s=100, label="観測点")

# 最適化された点のプロット
plt.scatter(res.x[0], res.x[1], color="green", s=100, label="最適化された点")

# 等高線のプロット
temp_mesh, humidity_mesh = np.meshgrid(
    np.linspace(0, 100, 100), np.linspace(0, 100, 100)
)
Z = np.zeros_like(temp_mesh)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i, j] = objective([temp_mesh[i, j], humidity_mesh[i, j]])
plt.contour(temp_mesh, humidity_mesh, Z, levels=20, alpha=0.5, colors="gray")

plt.xlabel("温度 (°C)")
plt.ylabel("湿度 (%)")
plt.title("温度と湿度の最適化")
plt.legend()
plt.grid(True)

# 結果のテキスト表示
plt.text(
    5,
    90,
    f"最適化された温度: {res.x[0]:.2f}°C\n最適化された湿度: {res.x[1]:.2f}%",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.show()

# 結果の表示
print(f"最適化された温度: {res.x[0]:.2f}°C")
print(f"最適化された湿度: {res.x[1]:.2f}%")
print(f"制約条件: {-constraint(res.x):.2f} >= 0")
