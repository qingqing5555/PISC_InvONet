import numpy as np

"""

edit: suzy 20231211

"""

# --------------------正演部分参数-----------------------------
peak_freq = 7.  # 频率
peak_source_time = 1 / peak_freq  # 周期
dx = 30.0  # 网格间距
dt = 0.003  # 时间间隔 0.006 0.003
nz = 100  # z 网格点 mar_small:100  over:94  mar_big:117  openfwi:70
ny = 310  # y 网格 mar_small:310   over: 400   mar_big:567
vmodel_dim = np.array([nz, ny])  # 速度模型网格

total_t = 6  # 采样时间s
nt = int(total_t / dt)  # 时间采样总维度
num_shots = 30  # shots次数  mar_small:30 over:39 mar_big:56

num_sources_per_shot = 1  # 每次1个震源
d_source = 10  # 其它：ny / (num_shots + 1) 310/31  * 30m = 300m  openfwi:400/40 *30=300m  10
first_source = 5  # 其它：ny / (num_shots + 1) 310/31  * 30m = 300m  openfwi:10
source_depth = 0  # 震源深度
source_spacing = np.floor(dx * ny / (num_shots + 1))  # 震源间距

num_receivers_per_shot = 310  # 接收器 310 400 567
d_receiver = 1  # 30m 1
first_receiver = 0  # 0m
receiver_depth = 0  # 接收器深度
receiver_spacing = np.floor(dx * ny / (num_receivers_per_shot + 1))  # 接收器间距


survey_pad = None
fix_value_depth = 0  # 把和真实模型一致的参数冻结

gfsigma = 'GS'  # 设置不同类型可以获得不同初始模型
if gfsigma == 'line':
    lipar = 1.0
elif gfsigma == 'lineminmax':
    lipar = 1.1
else:
    lipar = None

order = 4  # 正演精度 一般低就是4 高就是8 也可以设置更高
pml_width = [20, 20, 0, 20]   # 边界厚度[第一个维度的开始，第一个维度的最后，第二维度的开始，第二维度的最后]

AddNoise = True  # 增加噪声
if AddNoise == True:
    noise_type = 'Gaussian'
    noise_var = 20


# 滤波器（源） 仅允许某段频率通过
use_filter = False  # 开关
filter_type = 'highpass'  # 滤波器类型
freqmin = 3  # 最小频率
freqmax = None  # 最大频率
corners = 6  # corners
df = 1 / dt

learnAWGN = False  # 学习噪声等级

# --------------------传统反演参数-----------------------------
fwi_lr = 2  # 步长
fwi_batch = 30  # batches 不是深度学习里的batch 是deepwave传统反演里的
fwi_num_epochs = 100
fwi_weight_decay = 0  # 调节lr的参数，如0.1 （一般会设置lr逐渐减小 通过gamma和 step size）

if fwi_weight_decay > 0:
    fwi_stepsize = 100

AddTV = False  # ATV正则化项 防止过拟合
if AddTV:
    alpha_tv = 1. * 1e-4  # weight for ATV

fscale = 1e2  # 乘以归一化数据的标量，以便更好地反向传播(也可以通过改变学习率来实现)
data_norm = False  # normalizating
fwi_loss_type = 'L2'  # loss
savepoch = 50  # 50 epoch保存一次
plot_ite = 100  # 300 iteration输出一次图像

# -------------------- FWIGAN 参数 -----------------------------
gan_v_lr = 3 * 1e1  # 更新速度模型的 步长
gan_num_batches = 6  # 内存不够用 采样，隔+6采样一次，30 shot可以分成6块，每块5 shot（分成很多块）6*5  同理，可以根据自己模型的大小设置
start_epoch = 0
gan_num_epochs = 300
DFilter = 32  # 输入通道
leak_value = 0  # 激活函数的负斜率
gan_weight_decay = 0  # weight_decay参数可以设置参数在训练过程中的衰减，这和L2正则化的作用效果等价
if gan_weight_decay > 0:
    gan_stepsize = 100  # 隔100 step改变一下lr

gan_d_lr = 1 * 1e-3  # 鉴别器学习率

criticIter = 6  # 要能被shots整除 6;3
lamb = 10  # 梯度参数
