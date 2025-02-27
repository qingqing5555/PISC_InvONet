import os
from Config.Param import *

"""

edit: suzy 20231215

"""

# 设置文件路径
# os.chdir("/path/to/directory")
# 结果存储
main_dir = "../results/"
if not os.path.exists('../results/'):
    os.makedirs('../results/')
results_dir = main_dir

data_name = 'mar_smal_100_310'  # 这是存储文件名的一部分
ResultPath = results_dir + str(data_name) + '_dim' + str(vmodel_dim) + '_pf' + str(peak_freq) + '_dx' + str(
    dx) + '_dt' + \
             str(dt) + '_T' + str(total_t) + '_ns' + str(num_shots) + '_nrps' + str(num_receivers_per_shot) + \
             '_sig' + str(gfsigma)

if lipar != None:
    ResultPath = ResultPath + '_lip' + str(lipar)

if source_depth > 0:
    ResultPath = ResultPath + '_sd' + str(source_depth)

if receiver_depth > 0:
    ResultPath = ResultPath + '_rd' + str(receiver_depth)

if fix_value_depth > 0:
    ResultPath = ResultPath + '_fixv' + str(fix_value_depth)

if order != 4:
    ResultPath = ResultPath + '_order' + str(order)

if pml_width != 10 and pml_width != None:
    ResultPath = ResultPath + '_pml' + str(pml_width)

ResultPath = ResultPath + '/'
# 速度模型路径
data_path = "../data/mar_smal_100_310.bin"
