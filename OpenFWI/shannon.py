import math
import numpy as np
import cv2
from PIL import Image

def a2i(indata):
    mg = Image.new('L', indata.transpose().shape)
    mn = indata.min()
    a = indata - mn
    mx = a.max()
    a = a * 1. / mx
    return a

def count_elements(matrix):
    unique_elements, counts = np.unique(matrix, return_counts=True)
    return dict(zip(unique_elements, counts))

def shannon_entropy(data):
    # freq = collections.Counter(data)
    freq = count_elements(data)
    total = data.size
    # shannon_entropy = -sum([(freq[key]/total) * math.log(freq[key]/total, 2) for key in freq])
    shannonEnt = 0.0
    for key in freq:
        prob = float(freq[key])/total
        shannonEnt = shannonEnt -prob * math.log(prob, 2)
    return shannonEnt

def sobel_mean(data):
    data = a2i(data)
    sobelx = cv2.Sobel(data, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(data, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
    GSI = np.count_nonzero(gradient)/data.size
    result = np.sum(gradient)/data.size
    # sobel_mean = math.sqrt(np.dot(sobelx, sobelx) + np.dot(sobely, sobely))/data.size
    return result, GSI

# 导入数据 然后计算即可 SEG simulate openfwi都是代表数据集类型
# SEG
data = np.load('F:/suzy/OpenFWI/FWI/SEG/model1.npy')[0][0]
shannon = shannon_entropy(data)
print(shannon)
SI, GSI = sobel_mean(data)
print(SI)
print(GSI)

# simulate
# data = np.load('F:/suzy/open_data/Simulatedata/model1_1.npy')
# shannon = 0.0
# SI = 0
# GSI = 0
# for i in range(1, 500):
#     shannon += shannon_entropy(data[i][0])
#     si_tmp, gsi_tmp = sobel_mean(data[i][0])
#     SI += si_tmp
#     GSI += gsi_tmp
# print(shannon/500)
# print(SI/500)
# print(GSI/500)

# openfwi
# data = np.load('F:/suzy/open_data/OPENFWI/Curve_Vel_A/model/model60.npy')
#
# shannon = 0.0
# for i in range(1, 500):
#     shannon += shannon_entropy(data[i][0])
# print(shannon/500)
#
# SI = 0
# GSI = 0
# for i in range(1, 500):
#     shannon += shannon_entropy(data[i][0])
#     si_tmp, gsi_tmp = sobel_mean(data[i][0])
#     SI += si_tmp
#     GSI += gsi_tmp
# print(SI/500)
# print(GSI/500)

