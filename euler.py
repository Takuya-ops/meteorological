import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# パラメータ設定
n_lat = 50  # 緯度方向の格子点数
n_alt = 30  # 高度方向の格子点数
n_steps = 200  # シミュレーションのステップ数
dt = 0.1  # 時間ステップ（日）

# 格子の設定
lats = np.linspace(-90, 90, n_lat)
# print(lats)
alts = np.linspace(0, 30, n_alt)
# print(alts)
# breakpoint()
# raise Exception("ここで実行を停止")
lat_grid, alt_grid = np.meshgrid(lats, alts)
# print(lat_grid, alt_grid)


# 風速場の定義（ハドレー循環を簡略化）
def wind_field(lat, alt):
    v_lat = np.sin(np.radians(lat)) * np.cos(np.pi * alt / 30)  # 緯度方向の風
    # print(v_lat)
    v_alt = -np.cos(np.radians(lat)) * np.sin(np.pi * alt / 30)  # 高度方向の風
    print(v_alt)
    sys.exit()
    return v_lat, v_alt


# 初期のスカラー場（例：温度や湿度を表す）
scalar_field = np.zeros((n_alt, n_lat))


# アニメーションの更新関数
def update(frame):
    global scalar_field
    v_lat, v_alt = wind_field(lat_grid, alt_grid)

    # 移流方程式の数値解法（単純な前方差分法）
    d_scalar_lat = np.gradient(scalar_field, axis=1)
    d_scalar_alt = np.gradient(scalar_field, axis=0)

    scalar_field -= dt * (v_lat * d_scalar_lat + v_alt * d_scalar_alt)

    # ソース項（簡単な例として、赤道付近で加熱）
    source = (
        0.1 * np.exp(-(((lat_grid) / 15) ** 2)) * np.exp(-(((alt_grid - 15) / 5) ** 2))
    )
    scalar_field += dt * source

    # 境界条件の処理（単純な例）
    scalar_field[:, 0] = scalar_field[:, 1]  # 南極
    scalar_field[:, -1] = scalar_field[:, -2]  # 北極
    scalar_field[0, :] = scalar_field[1, :]  # 地表
    scalar_field[-1, :] = 0  # 上端（固定値）

    im.set_array(scalar_field)
    quiver.set_UVC(v_lat, v_alt)
    return im, quiver


# プロットの設定
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(
    scalar_field, cmap="RdBu_r", extent=[-90, 90, 0, 30], origin="lower", animated=True
)
quiver = ax.quiver(lat_grid, alt_grid, *wind_field(lat_grid, alt_grid), scale=50)
fig.colorbar(im, label="Scalar Value")
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Eulerian Meridional Circulation")

# アニメーションの作成
anim = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)
plt.show()
