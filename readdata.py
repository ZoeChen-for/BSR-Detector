
import matplotlib.pyplot as plt
import librosa
import numpy as np

# 导入数据
file = 'D:\\caeri\\data\\36rpm_Center_6kN Run 1\\LMS_3_Mout 02_-Y.wav'
sig, sr = librosa.load(file, sr=None)

plt.figure(1)
plt.plot(sig)

# 频域


# mfcc变换
mfccs = librosa.feature.mfcc(y = sig, sr = sr,n_mfcc=24)


print(mfccs.shape)
plt.figure(2)
plt.plot(mfccs)
plt.show()


