import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# 画像を読み込み、グレースケールに変換
image = Image.open("IMG_6042.PNG").convert("L")
img_array = np.array(image)

# 画像サイズの取得
height, width = img_array.shape

# 元の画像を表示
plt.figure(figsize=(10, 5))
plt.subplot(121)
# plt.imshow(img_array, cmap="gray")
# 青から黄色へのグラデーション
# plt.imshow(img_array, cmap="viridis")
# plt.imshow(img_array, cmap="coolwarm")
plt.imshow(img_array, cmap="jet")
# plt.imshow(img_array, cmap="RdBu")
plt.colorbar()
plt.colormaps()
plt.title("Original Image")
plt.axis("off")


# 2D フーリエ変換を適用
f_transform = np.fft.fft2(img_array)

# 低周波成分を中心に移動
f_shift = np.fft.fftshift(f_transform)

# 保持する成分の割合を設定（例：10%）
keep_fraction = 0.1

# 中心からの距離に基づいてマスクを作成
rows, cols = img_array.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), dtype=bool)
# print(mask)
# sys.exit()
r = int(np.sqrt(keep_fraction * rows * cols / np.pi))
# print(r)
# sys.exit()
y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r**2
mask[mask_area] = True

# マスクを適用
f_shift_compressed = f_shift * mask

# 逆シフトと逆フーリエ変換
f_inverse_shift = np.fft.ifftshift(f_shift_compressed)
img_back = np.fft.ifft2(f_inverse_shift)
img_back = np.abs(img_back)

# 圧縮後の画像を表示
plt.subplot(122)
plt.imshow(img_back, cmap="gray")
plt.title("Compressed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# 元のデータサイズ
original_size = img_array.size * img_array.itemsize

# 圧縮後のデータサイズ（非ゼロ要素のみ）
compressed_size = np.count_nonzero(f_shift_compressed) * f_shift_compressed.itemsize

# 圧縮率の計算
compression_ratio = (1 - compressed_size / original_size) * 100

print(f"圧縮率: {compression_ratio:.2f}%")
