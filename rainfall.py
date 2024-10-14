import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import japanize_matplotlib

plt.rcParams["font.family"] = "AppleGothic"


class PrecipitationModel:
    def __init__(self, initial_temp, initial_humidity, initial_pressure):
        self.temp = initial_temp  # 気温 [K]
        self.humidity = initial_humidity  # 相対湿度 [%]
        self.pressure = initial_pressure  # 気圧 [hPa]
        self.water_vapor = 0  # 水蒸気量 [kg/m^3]
        self.cloud_water = 0  # 雲水量 [kg/m^3]
        self.rain_water = 0  # 雨水量 [kg/m^3]

    def saturation_vapor_pressure(self, T):
        # Magnus公式による飽和水蒸気圧の計算
        return 6.1094 * np.exp(17.625 * (T - 273.15) / (T - 30.11))

    def update_water_vapor(self):
        # 相対湿度から水蒸気量を計算
        es = self.saturation_vapor_pressure(self.temp)
        self.water_vapor = self.humidity / 100 * es * 100 / (461.5 * self.temp)

    def condensation_rate(self):
        # 凝結率の計算
        es = self.saturation_vapor_pressure(self.temp)
        qs = es * 100 / (461.5 * self.temp)
        return max(0, (self.water_vapor - qs) / 100)  # 簡略化した凝結率

    def autoconversion_rate(self):
        # 自動転換率（雲水から雨水への変換）の計算
        return 0.001 * self.cloud_water if self.cloud_water > 0.0005 else 0

    def evaporation_rate(self):
        # 蒸発率の計算
        es = self.saturation_vapor_pressure(self.temp)
        qs = es * 100 / (461.5 * self.temp)
        return max(0, (qs - self.water_vapor) / 100)  # 簡略化した蒸発率

    def precipitation_rate(self):
        # 降水率の計算（単純化）
        return 0.1 * self.rain_water

    def model(self, y, t):
        self.water_vapor, self.cloud_water, self.rain_water = y

        condensation = self.condensation_rate()
        autoconversion = self.autoconversion_rate()
        evaporation = self.evaporation_rate()
        precipitation = self.precipitation_rate()

        dWV_dt = -condensation + evaporation
        dCW_dt = condensation - autoconversion
        dRW_dt = autoconversion - precipitation

        return [dWV_dt, dCW_dt, dRW_dt]

    def simulate(self, time_span):
        y0 = [self.water_vapor, self.cloud_water, self.rain_water]
        t = np.linspace(0, time_span, 1000)
        solution = odeint(self.model, y0, t)

        self.water_vapor, self.cloud_water, self.rain_water = solution[-1]
        return t, solution


# モデルの初期化と実行
model = PrecipitationModel(
    initial_temp=288, initial_humidity=80, initial_pressure=1013.25
)
model.update_water_vapor()

time_span = 3600  # 1時間のシミュレーション
t, solution = model.simulate(time_span)

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], label="Water Vapor")
plt.plot(t, solution[:, 1], label="Cloud Water")
plt.plot(t, solution[:, 2], label="Rain Water")
plt.xlabel("Time (s)")
plt.ylabel("Water Content (kg/m^3)")
plt.title("Precipitation Process Simulation")
plt.legend()
plt.grid(True)
plt.show()

# 総降水量の計算
total_precipitation = np.trapz(model.precipitation_rate() * solution[:, 2], t)
print(f"Total precipitation over {time_span/3600} hours: {total_precipitation:.2f} mm")
