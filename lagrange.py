import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# パラメータ設定
n_particles = 1000  # 粒子の数
n_steps = 500  # シミュレーションのステップ数
dt = 0.1  # 時間ステップ
latitude_range = (-90, 90)  # 緯度範囲（度）
altitude_range = (0, 30)  # 高度範囲（km）


# 風速場の定義（ハドレー循環を簡略化）
def wind_field(lat, alt):
    v_lat = np.sin(np.radians(lat)) * np.cos(np.pi * alt / 30)  # 緯度方向の風
    v_alt = -np.cos(np.radians(lat)) * np.sin(np.pi * alt / 30)  # 高度方向の風
    return v_lat, v_alt


# 粒子の初期位置
particles = np.random.uniform(
    low=[latitude_range[0], altitude_range[0]],
    high=[latitude_range[1], altitude_range[1]],
    size=(n_particles, 2),
)


# アニメーションの更新関数
def update(frame):
    global particles
    for i in range(n_particles):
        v_lat, v_alt = wind_field(particles[i, 0], particles[i, 1])
        particles[i, 0] += v_lat * dt
        particles[i, 1] += v_alt * dt

        # 境界条件の処理
        particles[i, 0] = np.clip(particles[i, 0], latitude_range[0], latitude_range[1])
        particles[i, 1] = np.clip(particles[i, 1], altitude_range[0], altitude_range[1])

    scatter.set_offsets(particles)
    return (scatter,)


# プロットの設定
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.5)
ax.set_xlim(latitude_range)
ax.set_ylim(altitude_range)
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Simplified Lagrangian Meridional Circulation")

# アニメーションの作成
anim = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)
plt.show()
