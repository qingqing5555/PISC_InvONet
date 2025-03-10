from Functions.data_load import *
from Functions.my_math import *
from Functions.data_plot import *
from Config.Path import *
import torch.optim as optim
import time

"""

This file is a deep learning implementation of the traditional inversion method

"""


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# ---------------------------Load data------------------------------
# 1 Speed Model
model_true = loadtruemodel(data_path, vmodel_dim).to(device)
# model_true = torch.Tensor(np.load('F:/suzy/OpenFWI/FWI/SEG/model1.npy')[0][0])

# 2 Initial Model
init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('There is an already built initial model, loading...')
    model = load_init_model(init_model_path).clone().to(device)
    model.requires_grad = True
    print(' Real model maximum wave speed:', model_true.max())
    # Set the model parameters as trainable here, where model is a clone of the initial model
    model = torch.nn.Parameter(model)
else:
    raise Exception('No initial model data exists, please build...')

# 3 Create an array containing source and receiver positions
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)
x_s, x_r = x_s.to(device), x_r.to(device)

# 4 Seismic Source
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('Initial source exists, loading...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
    source_amplitudes_init.to(device)
    source_amplitudes_true.to(device)
else:
    raise Exception('No initial source exists, please construct...')

# 5 Seismic data
#If there is noise
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('Earthquake data (noise) already exists, loading...')
        receiver_amplitudes_true = loadrcv(noise_filepath).to(device)
    else:
        raise Exception('No seismic data (noise) exists, please build...')
else:
    # If there is no noise
    rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
    if os.path.exists(rcv_filepath):
        print('Earthquake data (clean) already exists, loading...')
        receiver_amplitudes_true = loadrcv(rcv_filepath).to(device)
    else:
        raise Exception('No seismic data (clean) exists, please build...')

# --------------------------Hyperparameters-----------------------------------
# Optimizer: training parameters, step size, momentum parameters, numerical stability parameters, L2 regularization hyperparameters
# my_model = ConvModNet_SEG()
# optimizer = torch.optim.AdamW(model.parameters(), lr=fwi_lr, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer = optim.Adam([{'params': model, 'lr': fwi_lr, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0}])

if fwi_weight_decay > 0:
    # Optimizer object, how many times to update the learning rate per cycle, each update of lr's gamma times
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=fwi_stepsize, gamma=fwi_weight_decay)
# Loss function selection loss option: L1, L2, 1-D W1
if fwi_loss_type == 'L1':
    criterion = torch.nn.L1Loss()
elif fwi_loss_type == 'L2':
    criterion = torch.nn.MSELoss()
elif fwi_loss_type == 'W1':
    trans_type = 'linear'  # linear, square, exp, softplus, abs
else:
    raise NotImplementedError

# Result storage path - Loss function type-step-length-batch-normalization-epoch number
fwi_result = ResultPath + 'FWI' + '_loss' + str(fwi_loss_type) + '_lr' + str(fwi_lr) + \
             '_batch' + str(fwi_batch) + '_norm' + str(data_norm) + '_epoch' + str(fwi_num_epochs)

if fwi_weight_decay > 0:
    fwi_result = fwi_result + '_dra' + str(fwi_weight_decay) + '_step' + str(fwi_stepsize)

if AddTV:
    fwi_result = fwi_result + '_alp' + str(alpha_tv)

fwi_result = fwi_result + '/'

if not os.path.exists(fwi_result):
    os.makedirs(fwi_result)

rcv_amps_true = receiver_amplitudes_true.clone()

print()
print('*******************************************')
print('          START Traditional FWI            ')
print('*******************************************')
# 初始化参数
SNR = 0.0
SSIM = 0.0
Loss = 0.0
ERROR = 0.0
TOL = 0.0


def fwi_main():
    global model_true, source_amplitudes_init, SNR, SSIM, Loss, ERROR

    t_start = time.time()
    model_true = model_true.view(nz, ny)
    # How many shots per batch?
    num_shots_per_batch = int(num_shots / fwi_batch)

    for i in range(fwi_num_epochs):
        # loss initialization
        epoch_loss = 0.0

        for it in range(fwi_batch):
            iteration = i * fwi_batch + it + 1
            optimizer.zero_grad()  

            # prop = deepwave.scalar.Propagator({'vp': model}, dx, pml_width, order, survey_pad)
            batch_src_amps = source_amplitudes_init.repeat(1, num_shots_per_batch, 1).to(device) 
            batch_rcv_amps_true = rcv_amps_true[:, it::fwi_batch].to(device)  # Sampled real seismic data

            batch_x_s = x_s[it::fwi_batch].to(device)
            batch_x_r = x_r[it::fwi_batch].to(device)
            # batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)  # Simulating wave propagation
            batch_rcv_amps_pred = createdata(model, dx, dt, batch_src_amps, batch_x_s, batch_x_r,
                                             order, pml_width, peak_freq)

            if fwi_loss_type == 'L1' or fwi_loss_type == 'L2':
                if data_norm:
                    # normalize
                    batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1, 0, 2).unsqueeze(1)).squeeze(
                        1).permute(1, 0, 2) * fscale
                    batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1, 0, 2).unsqueeze(1)).squeeze(
                        1).permute(1, 0, 2) * fscale
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
                else:
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
            elif fwi_loss_type == 'W1':
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true, trans_type, theta=1.1)
            else:
                raise NotImplementedError

            if fix_value_depth > 0:
                fix_model_grad(fix_value_depth, model)

            epoch_loss += loss.item()
            loss.backward()

          
            torch.nn.utils.clip_grad_value_(model, 1e3)

            optimizer.step()

           
            model.data = torch.clamp(model.data, min=1e-12)

        if fwi_weight_decay > 0:
            scheduler.step()

        print('Epoch:', i + 1, 'Loss: ', epoch_loss / fwi_batch)
        Loss = np.append(Loss, epoch_loss / fwi_batch)

       
        snr = ComputeSNR(model.detach().cpu().numpy(),
                         model_true.detach().cpu().numpy())
        SNR = np.append(SNR, snr)

        ssim = ComputeSSIM(model.detach().cpu().numpy(),
                           model_true.detach().cpu().numpy())
        SSIM = np.append(SSIM, ssim)

        rerror = ComputeRE(model.detach().cpu().numpy(),
                           model_true.detach().cpu().numpy())
        ERROR = np.append(ERROR, rerror)

        if iteration % plot_ite == 0:
            plotcomparison(gt=model_true.cpu().data.numpy(),
                           pre=model.cpu().data.numpy(),
                           ite=iteration, SaveFigPath=fwi_result)
            # plot Loss
            PlotFWILoss(loss=Loss, SaveFigPath=fwi_result)

            # plot SNR, ERROR, and SSIM
            PlotSNR(SNR=SNR, SaveFigPath=fwi_result)
            PlotSSIM(SSIM=SSIM, SaveFigPath=fwi_result)
            PlotERROR(ERROR=ERROR, SaveFigPath=fwi_result)

        if (i + 1) % savepoch == 0 or (i + 1) == fwi_num_epochs:
          
            spio.savemat(fwi_result + 'FWIRec_' + str(fwi_loss_type) + '.mat',
                         {'rec': model.cpu().data.numpy()})
            spio.savemat(fwi_result + 'FWIMetric_' + str(fwi_loss_type) + '.mat',
                         {'SNR': SNR, 'SSIM': SSIM,
                          'Loss': Loss, 'ERROR': ERROR})

    t_end = time.time()
    elapsed_time = t_end - t_start
    print('Running complete in {:.0f}m  {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    np.savetxt(fwi_result + 'run_result.txt',
               np.hstack((fwi_num_epochs, elapsed_time // 60, elapsed_time % 60, snr, ssim, rerror)),
               fmt='%5.4f')  # ssim,


if __name__ == "__main__":
    fwi_main()
    exit(0)
