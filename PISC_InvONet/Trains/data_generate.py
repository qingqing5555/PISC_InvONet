from Functions.data_load import *
from Functions.data_plot import *
from Config.Path import *

"""
This file can generate corresponding seismic data based on the [speed model path, initial model type, noise parameters, forward parameters] set in the Path

"""

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# --------------------------Generate data---------------------------------------------------------#
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

model_true = loadtruemodel(data_path, vmodel_dim)
# model_true = torch.Tensor(np.load('../../OpenFWI/FWI/SEG/model1.npy')[0][0])

if not os.path.exists(ResultPath):
    os.makedirs(ResultPath)
init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('The initial model already exists..')
else:
    print('Preparing to build the initial model....')
    model_init = createInitialModel(model_true, gfsigma, lipar, fix_value_depth)
    # model = model_init.clone()
    vmin, vmax = np.percentile(model_true.cpu().data.numpy(), [2, 98])  # Calculate the arbitrary percentile of a multidimensional array
    plotinitmodel(model_init, model_true, vmin, vmax, ResultPath)
    spio.savemat(init_model_path,
                 {'initmodel': model_init.cpu().data.numpy()})
    print('Initial model saved successfully..')

# Build seismic source, earthquake data
# Seismic Source
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('Initial earthquake source already exists..')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
else:
    #Generate seismic source
    print('Generating seismic source data...')
    source_amplitudes_true = createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots)
    source_amplitudes_true = torch.from_numpy(source_amplitudes_true)
    # If using a filter
    if use_filter:
        source_amplitudes_filt = createFilterSourceAmp(peak_freq, nt, dt, peak_source_time,
                                                       num_shots,
                                                       use_filter, filter_type, freqmin, freqmax,
                                                       corners, df)

        source_amplitudes_filt = torch.from_numpy(source_amplitudes_filt)
        #Get initial filter source
        source_amplitudes_init = source_amplitudes_filt[:, 0, 0].reshape(-1, 1, 1)
    else:
        #Get initial earthquake source
        source_amplitudes_init = source_amplitudes_true[:, 0, 0].reshape(-1, 1, 1)
    spio.savemat(init_source_filepath,
                 {'initsource': source_amplitudes_init.cpu().data.numpy(),
                  'truesource': source_amplitudes_true.cpu().data.numpy()})

    plotinitsource(init=source_amplitudes_init.cpu().detach().numpy(),
                   gt=source_amplitudes_true[:, 0, 0].cpu().detach().numpy(), SaveFigPath=ResultPath)
    plotsourcespectra(init_source=source_amplitudes_init.cpu().detach().numpy(),
                      true_source=source_amplitudes_true[:, 0, 0].cpu().detach().numpy(),
                      SaveFigPath=ResultPath)
#Seismic data
rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
if os.path.exists(rcv_filepath):
    print('Earthquake data already exists...')
    receiver_amplitudes_true = loadrcv(rcv_filepath)
    plotoneshot(receiver_amplitudes_true, ResultPath)
else:
    print('Generating seismic data.........')
    # Generate seismic data
    #  over x_s=10... x_r=0...399   mar_big:x_s=110...560  x_r=0...566
    x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)
    model_true = model_true.to(device)
    source_amplitudes_true = source_amplitudes_true.to(device)
    receiver_amplitudes = createdata(model_true, dx, dt, source_amplitudes_true, x_s, x_r,
               order, pml_width, peak_freq)
    receiver_amplitudes_true = receiver_amplitudes
    plotoneshot(receiver_amplitudes_true, ResultPath)
    # save the receiver amplitudes
    spio.savemat(rcv_filepath,
                 {'true': receiver_amplitudes_true.cpu().data.numpy(),
                  'rcv': receiver_amplitudes.cpu().data.numpy()})

# [nt, num_shots, num_receivers_per_shot]
print('Receiver shape：', receiver_amplitudes_true.shape)
#  Add noise
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    if not os.path.exists(ResultPath):
        os.makedirs(ResultPath)
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('Noisy receiver amplitudes exists, downloading......')
        # here the rcv_amps_true is contaminated by noise
        receiver_amplitudes_true = loadrcv(noise_filepath)

    else:
        print('Generating noisy seismic data.......\n',
              'Add %.2f Gaussian noise' % (noise_var))
        receiver_amplitudes_noise = add_noise(receiver_amplitudes_true, noise_type, noise_var)
        # Calculate the average SNR of the noise
        noise_snr = ComputeSNR(receiver_amplitudes_noise.permute(1, 0, 2),
                               receiver_amplitudes_true.permute(1, 0, 2))
        print('The noise level is %.2f dB' % noise_snr)
        #Print noise version data
        plotoneshot(receiver_amplitudes_noise, ResultPath)
        # save the receiver amplitudes
        spio.savemat(noise_filepath,
                     {'true': receiver_amplitudes_noise.cpu().data.numpy(),
                      'rcv': receiver_amplitudes_true.cpu().data.numpy()})
print('Data generation successful~')
