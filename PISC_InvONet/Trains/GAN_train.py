from Functions.data_load import *
from Functions.data_plot import *
from Functions.my_math import *
from Config.Path import *
from Models.PhySimulator import *
from Models.Discriminator import *
from torch import optim
import time

"""

这个文件是FWIGAN的实现 详情可以去找原文

"""

# 检查cudnn cuda
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
if cuda_available:
    print("use cuda")
else:
    print("use cpu")

# ----------------------------加载数据------------------------------
# 1 速度模型
model_true = loadtruemodel(data_path, vmodel_dim).to(device)

# 2 初始模型
init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('存在已构建的初始模型，正在加载...')
    model = load_init_model(init_model_path).clone().to(device)
    model.requires_grad = True
    print(' 真实模型最大波速: ', model_true.max())
    # 把model设置为可训练的参数 这里model是初始模型的clone
    model = torch.nn.Parameter(model)
else:
    raise Exception('不存在初始模型数据，请构建...')

# 3 创建包含源和接收器位置的数组
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)

# 4 震源
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('初始震源已存在,正在加载...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
    source_amplitudes_init = source_amplitudes_init.to(device)
    source_amplitudes_true = source_amplitudes_true.to(device)
else:
    raise Exception('不存在初始震源，请构建...')

# 5 地震数据
# 如果有噪声
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('地震数据（noise）已存在,正在加载...')
        rcv_amps_true = loadrcv(noise_filepath).to(device)
    else:
        raise Exception('不存在地震数据（noise），请构建...')
else:
    # 如果无噪声
    rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
    if os.path.exists(rcv_filepath):
        print('地震数据（clean）已存在,正在加载...')
        rcv_amps_true = loadrcv(rcv_filepath).to(device)
    else:
        raise Exception('不存在地震数据（clean），请构建...')

# ----------------------物理生成器和判别器-----------------------------#
PhySimulator = PhySimulator(dx, num_shots, gan_num_batches,
                            x_s, x_r, dt, pml_width,
                            order, peak_freq).to(device)

num_shots_per_batch = int(num_shots / gan_num_batches)  # 每个batch包含的shots数量
Filters = np.array([DFilter, 2 * DFilter, 4 * DFilter, 8 * DFilter, 16 * DFilter, 32 * DFilter], dtype=int)
netD = Discriminator(batch_size=num_shots_per_batch, ImagDim=[nt, num_receivers_per_shot],
                     LReLuRatio=0.1, filters=Filters,
                     leak_value=leak_value)
# 初始化
netD.apply(lambda m: weights_init(m, leak_value))
netD = netD.to(device)
optim_d = optim.Adam(netD.parameters(), lr=gan_d_lr, betas=(0.5, 0.9),
                     eps=1e-8, weight_decay=0)
optim_g = optim.Adam([{'params': model, 'lr': gan_v_lr, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0}])

# 学习率调节器
if gan_weight_decay > 0:
    scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g,
                                                  step_size=gan_stepsize,
                                                  gamma=gan_weight_decay)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d,
                                                  step_size=gan_stepsize,
                                                  gamma=gan_weight_decay)
# 如果是噪声数据，学习噪声
if AddNoise == True and noise_var != None and learnAWGN == True:
    init_snr_guess = np.array(20, dtype="float32")
    gan_noi_lr = 1e-2
    learn_snr, learn_snr_init = createlearnSNR(init_snr_guess, device)
    learn_snr = torch.nn.Parameter(learn_snr)
    optim_noi = optim.Adam(
        [{'params': learn_snr, 'lr': gan_noi_lr, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0}])
    if gan_weight_decay > 0:
        scheduler_noi = torch.optim.lr_scheduler.StepLR(optim_noi,
                                                        step_size=gan_stepsize,
                                                        gamma=gan_weight_decay)
else:
    learn_snr = None

# 计算参数数量
s = sum(np.prod(list(p.size())) for p in netD.parameters())
print('判别器参数数量为: %d' % s)

# 保存路径
gan_result = ResultPath + 'GAN' + '_vlr' + str(gan_v_lr) + '_dlr' + str(gan_d_lr) + '_batch' + str(
    gan_num_batches) + '_epoch' + str(gan_num_epochs) + '_criIte' + str(criticIter)

if gan_weight_decay > 0:
    gan_result = gan_result + '_dra' + str(gan_weight_decay) + '_step' + str(gan_stepsize)

if lamb != 10:
    gan_result = gan_result + '_lamb' + str(lamb)

if AddNoise is True and noise_var is not None and learnAWGN is True:
    gan_result = gan_result + '_noilr' + str(gan_noi_lr) + '_initsnr' + str(init_snr_guess)

gan_result = gan_result + '/'

if not os.path.exists(gan_result):
    os.makedirs(gan_result)

print('*******************************************')
print('             START GAN                  ')
print('*******************************************')

# ----------超参数----------- #
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
one = one.to(device)
mone = mone.to(device)
DLoss = 0.0
WDist = 0.0
GLoss = 0.0
DiscReal = 0.0
DiscFake = 0.0
SNR = 0.0
SSIM = 0.0
ERROR = 0.0


def fwi_gan_main():
    global DLoss, WDist, GLoss, one, mone, DiscReal, DiscFake, model_true, SNR, SSIM, ERROR

    start_time = time.time()
    # vmin, vmax = np.percentile(model_true.detach().cpu().numpy(), [2, 98])
    model_true = model_true.view(nz, ny)

    for epoch in range(start_epoch, gan_num_epochs):
        print("Epoch: " + str(epoch + 1))

        for it in range(gan_num_batches):
            iteration = epoch * gan_num_batches + it + 1
            # -------------------判别器训练------------------------ #
            for p in netD.parameters():  # 每轮重置判别器参数grad设置，训练时为true
                p.requires_grad_(True)  # 训练生成器的时候是false
            for j in range(criticIter):
                netD.train()
                netD.zero_grad()
                if it * criticIter + j < gan_num_batches:
                    # 取出真实数据
                    batch_rcv_amps_true = rcv_amps_true[:, it * criticIter + j::gan_num_batches]
                else:
                    batch_rcv_amps_true = rcv_amps_true[:, ((it * criticIter + j) % gan_num_batches)::gan_num_batches]

                d_real = batch_rcv_amps_true.detach()  # 别名，真实数据
                with torch.no_grad():
                    model_fake = model  # 初始模型
                    if AddNoise is True and noise_var is not None and learnAWGN is True:
                        learn_snr_fake = learn_snr.detach()
                    else:
                        learn_snr_fake = None
                # 模拟得到正演数据
                d_fake = PhySimulator(model_fake,
                                      source_amplitudes_init, it, criticIter, j,
                                      'TD', AddNoise, learn_snr_fake, learnAWGN)

                # 打印
                if j == criticIter - 1 and iteration % plot_ite == 0:
                    plotfakereal(fakeamp=d_fake[:, 0].cpu().data.numpy(),
                                 realamp=d_real[:, 0].cpu().data.numpy(),
                                 ite=iteration, cite=j + 1, SaveFigPath=gan_result)

                # 真实数据放到判别器得到模型
                # 改一下维度格式 [nt,num_shots,nnum_receiver] -> [num_shots,nt,num_receiver] 5 2000 310
                d_real = d_real.permute(1, 0, 2)
                # 再改一下[num_shots,1,nt,num_receiver] 5 1 2000 310
                d_real = d_real.unsqueeze(1)
                disc_real = netD(d_real)
                disc_real = disc_real.mean()

                # 模拟数据放到判别器得到模型
                d_fake = d_fake.permute(1, 0, 2)
                d_fake = d_fake.unsqueeze(1)
                disc_fake = netD(d_fake)
                disc_fake = disc_fake.mean()

                # 累计真实和模拟数据的分数
                DiscReal = np.append(DiscReal, disc_real.item())
                DiscFake = np.append(DiscFake, disc_fake.item())

                # 梯度惩罚
                gradient_penalty = calc_gradient_penalty(netD, d_real, d_fake,
                                                         batch_size=num_shots_per_batch,
                                                         channel=1, lamb=lamb,
                                                         device=device)
                # 计算总得分
                disc_cost = disc_fake - disc_real + gradient_penalty
                print('Epoch: %03d  Ite: %05d  DLoss: %f' % (epoch + 1, iteration, disc_cost.item()))

                w_dist = -(disc_fake - disc_real)
                # 累积总得分和 w_dist？
                DLoss = np.append(DLoss, disc_cost.item())
                WDist = np.append(WDist, w_dist.item())

                # 根据总得分进行反向传播
                disc_cost.backward()

                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(netD.parameters(),
                                               1e3)  # 1e3 for smoothed initial model; 1e6 for linear model

                # optimize
                optim_d.step()

                # scheduler for discriminator
                if gan_weight_decay > 0:
                    scheduler_d.step()

                # -----------------可视化---------------#
                if j == criticIter - 1 and iteration % plot_ite == 0:
                    PlotDLoss(dloss=DLoss, wdist=WDist,
                              SaveFigPath=gan_result)

                    # ------------模型更新--------------#
            for p in netD.parameters():
                p.requires_grad_(False)  # 这次把判别器冻结

            for k in range(1):
                optim_g.zero_grad()
                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    optim_noi.zero_grad()
                # 生成假的数据
                g_fake = PhySimulator(model,
                                      source_amplitudes_init, it, criticIter, j,
                                      'TG', AddNoise, learn_snr, learnAWGN)

                # 改改维度格式
                g_fake = g_fake.permute(1, 0, 2)
                g_fake = g_fake.unsqueeze(1)

                if fix_value_depth > 0:
                    fix_model_grad(fix_value_depth, model)

                # 计算生成器loss
                gen_cost = netD(g_fake)
                gen_cost = gen_cost.mean()
                gen_cost.backward(mone)
                gen_cost = - gen_cost
                print('Epoch: %03d  Ite: %05d  GLoss: %f' % (epoch + 1, iteration, gen_cost.item()))
                GLoss = np.append(GLoss, gen_cost.item())

                # 梯度裁剪 超过阈值就进行裁剪
                torch.nn.utils.clip_grad_value_(model,
                                                1e3)  # mar_smal: 1e1 smoothed initial model; 1e3 for linear model

                # optimize
                optim_g.step()

                # 生成器学习率调整
                if gan_weight_decay > 0:
                    scheduler_g.step()

                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    # Clips gradient value of learn_snr if needed
                    torch.nn.utils.clip_grad_value_(learn_snr, 1e1)
                    # optimize
                    optim_noi.step()
                    if gan_weight_decay > 0:
                        scheduler_noi.step()

                # clip 保证值大于0
                model.data = torch.clamp(model.data, min=1e-12)

                # compute the SNR, SSIM and realtive error between GT and inverted model
                snr = ComputeSNR(model.detach().cpu().numpy(),
                                 model_true.detach().cpu().numpy())
                SNR = np.append(SNR, snr)

                ssim = ComputeSSIM(model.detach().cpu().numpy(),
                                   model_true.detach().cpu().numpy())
                SSIM = np.append(SSIM, ssim)

                rerror = ComputeRE(model.detach().cpu().numpy(),
                                   model_true.detach().cpu().numpy())
                ERROR = np.append(ERROR, rerror)

            # ---------------可视化---------------------#
            if iteration % plot_ite == 0:
                # 输出真实模型和现在的反演结果
                plotcomparison(gt=model_true.cpu().data.numpy(),
                               pre=model.cpu().data.numpy(),
                               ite=iteration, SaveFigPath=gan_result)
                # 输出GLoss
                PlotGLoss(gloss=GLoss, SaveFigPath=gan_result)

                # 输出SNR, RSNR, SSIM and ERROR
                PlotSNR(SNR=SNR, SaveFigPath=gan_result)
                PlotSSIM(SSIM=SSIM, SaveFigPath=gan_result)
                PlotERROR(ERROR=ERROR, SaveFigPath=gan_result)
                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    print('learned snr:', learn_snr)

        if (epoch + 1) % savepoch == 0 or (epoch + 1) == gan_num_epochs:
            # ------------------保存结果-----------------#
            spio.savemat(gan_result + 'GANRec' + '.mat',
                         {'rec': model.cpu().data.numpy()})
            spio.savemat(gan_result + 'GANMetric' + '.mat',
                         {'SNR': SNR, 'SSIM': SSIM, 'ERROR': ERROR,
                          'DLoss': DLoss, 'WDist': WDist, 'GLoss': GLoss})

            # ----------------------保存模型----------------------#####
        if (epoch + 1) % savepoch == 0 or (epoch + 1) == gan_num_epochs:
            if gan_weight_decay > 0:
                dis_state = {
                    'epoch': epoch + 1,
                    'state_dict': netD,
                    'optim_d': optim_d,
                    'scheduler_d': scheduler_d,
                }
                torch.save(dis_state, gan_result + "netD.pth.tar")


            elif gan_weight_decay == 0:
                dis_state = {
                    'epoch': epoch + 1,
                    'state_dict': netD,
                    'optim_d': optim_d,
                }
                torch.save(dis_state, gan_result + "netD.pth.tar")

            else:
                raise NotImplementedError

    time_elapsed = time.time() - start_time
    print('训练总时长 {:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # record the consuming time
    np.savetxt(gan_result + 'run_result.txt',
               np.hstack((epoch + 1, time_elapsed // 60, time_elapsed % 60, snr, ssim, rerror)), fmt='%3.4f')
    if AddNoise is True and noise_var is not None and learnAWGN is True:
        np.savetxt(gan_result + 'run_result.txt', np.hstack(
            (epoch + 1, time_elapsed // 60, time_elapsed % 60, snr, ssim, rerror, learn_snr.cpu().data.numpy())),
                   fmt='%3.4f')


# ----------  Run Code ---------------------- #
if __name__ == "__main__":
    fwi_gan_main()
    exit(0)
