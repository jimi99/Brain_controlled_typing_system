import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 设置数据目录路径
data_dir = r'E:\bigproject\离线实验\raw_data\光电池822'

# 获取所有CNT文件
cnt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.cnt')]

# 分组文件，假设文件名中的数字代表分组
groups = {}
for file in cnt_files:
    group_key = int(os.path.basename(file).split('_')[0])
    if group_key not in groups:
        groups[group_key] = []
    groups[group_key].append(file)

# 读取数据、计算频率、打印频率并绘制波形和频谱
for group, files in groups.items():
    print(f"\nGroup {group}:")
    for file in files:
        raw = mne.io.read_raw_cnt(file, preload=True)  # 读取CNT文件

        # 检查'CPZ'通道是否存在
        if 'CPZ' in raw.ch_names:
            # 获取'CPZ'通道的数据
            picks = mne.pick_channels(raw.ch_names, include=['CPZ'])
            data = raw.get_data(picks=picks)  # 获取数据

            # 确保times数组的形状与data的样本维度一致
            times = raw.times[:data.shape[1]]

            # 绘制'CPZ'通道的时域波形
            plt.figure()
            plt.title('Time Domain Waveform of CPZ Channel')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.plot(times, data[0])  # 绘制第一个通道的数据
            plt.xlim(0, 2)  # 设置x轴的范围为0-10秒
            plt.show()

            # # 使用FFT来估计频率
            # fft_values = fft(data[0])  # 对第一个通道的数据进行FFT
            # n_samples = data.shape[1]
            # sampling_rate = raw.info['sfreq']
            # freqs = fftfreq(n=n_samples, d=1/sampling_rate)
            # # 找到峰值频率，忽略直流分量（FFT的第一个值）
            # peak_freq_idx = np.argmax(np.abs(fft_values[1:n_samples//2]))
            # peak_freq = freqs[peak_freq_idx]
            # print(f"File {file}: Peak Frequency {np.round(peak_freq, 2)} Hz")
            #
            # # 绘制频率谱
            # plt.figure()
            # plt.title('Frequency Spectrum of CPZ Channel')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.plot(freqs[1:n_samples//2], np.abs(fft_values[1:n_samples//2]))
            # plt.xlim(0, sampling_rate / 2)  # 限制x轴显示频率范围至奈奎斯特频率
            # plt.xlim(0, 200)
            # plt.show()
        else:
            print(f"'CPZ' channel not found in file {file}.")