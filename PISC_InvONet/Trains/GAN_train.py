from Functions.data_load import *
from Functions.data_plot import *
from Functions.my_math import *
from Config.Path import *
from Models.PhySimulator import *
from Models.Discriminator import *
from torch import optim
import time



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
if cuda_available:
    print("use cuda")
else:
    print("use cpu")

# ----------------------------Load data------------------------------

model_true = loadtruemodel(data_path, vmodel_dim).to(device)


init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('There is an already built initial model, loading...')
    model = load_init_model(init_model_path).clone().to(device)
    model.requires_grad = True
    print(' Real model maximum wave speed: ', model_true.max())
    # Set the model parameters as trainable here, where model is a clone of the initial model
    model = torch.nn.Parameter(model)
else:
    raise Exception('No initial model data exists, please build...')

# Create an array containing source and receiver positions
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)

# Seismic Source
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('Initial earthquake source already exists, loading...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
    source_amplitudes_init = source_amplitudes_init.to(device)
    source_amplitudes_true = source_amplitudes_true.to(device)
else:
    raise Exception('No initial earthquake source exists, please build...')

# Seismic data
#If there is noise
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('Earthquake data (noise) already exists, loading....')
        rcv_amps_true = loadrcv(noise_filepath).to(device)
    else:
        raise Exception('No seismic data (noise) exists, please build...')
else:
    # If there is no noise
    rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
    if os.path.exists(rcv_filepath):
        print('Earthquake data (clean) already exists, loading...')
        rcv_amps_true = loadrcv(rcv_filepath).to(device)
    else:
        raise Exception('No seismic data (clean) exists, please build...')

# ----------------------Physical generator and discriminator-----------------------------#
PhySimulator = PhySimulator(dx, num_shots, gan_num_batches,
                            x_s, x_r, dt, pml_width,
                            order, peak_freq).to(device)

num_shots_per_batch = int(num_shots / gan_num_batches)  # Number of shots per batch
Filters = np.array([DFilter, 2 * DFilter, 4 * DFilter, 8 * DFilter, 16 * DFilter, 32 * DFilter], dtype=int)
netD = Discriminator(batch_size=num_shots_per_batch, ImagDim=[nt, num_receivers_per_shot],
                     LReLuRatio=0.1, filters=Filters,
                     leak_value=leak_value)
# Initialize
netD.apply(lambda m: weights_init(m, leak_value))
netD = netD.to(device)
optim_d = optim.Adam(netD.parameters(), lr=gan_d_lr, betas=(0.5, 0.9),
                     eps=1e-8, weight_decay=0)
optim_g = optim.Adam([{'params': model, 'lr': gan_v_lr, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0}])

# Learning rate regulator
if gan_weight_decay > 0:
    scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g,
                                                  step_size=gan_stepsize,
                                                  gamma=gan_weight_decay)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d,
                                                  step_size=gan_stepsize,
                                                  gamma=gan_weight_decay)
# If it is noise data, learn the noise
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

#Calculate parameter quantity
s = sum(np.prod(list(p.size())) for p in netD.parameters())
print('The number of discriminator parameters is:%d' % s)

#Save Path
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
print('             START                  ')
print('*******************************************')

# ----------Hyperparameters----------- #
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
            # -------------------Discriminator training------------------------ #
            for p in netD.parameters():  # Each round resets the discriminator parameter grad setting, set to true during training
                p.requires_grad_(True)  # Training the generator is false
            for j in range(criticIter):
                netD.train()
                netD.zero_grad()
                if it * criticIter + j < gan_num_batches:
                    # Extract real data
                    batch_rcv_amps_true = rcv_amps_true[:, it * criticIter + j::gan_num_batches]
                else:
                    batch_rcv_amps_true = rcv_amps_true[:, ((it * criticIter + j) % gan_num_batches)::gan_num_batches]

                d_real = batch_rcv_amps_true.detach()  # Alias, real data
                with torch.no_grad():
                    model_fake = model  #Initial Model
                    if AddNoise is True and noise_var is not None and learnAWGN is True:
                        learn_snr_fake = learn_snr.detach()
                    else:
                        learn_snr_fake = None
                # Simulate forward data
                d_fake = PhySimulator(model_fake,
                                      source_amplitudes_init, it, criticIter, j,
                                      'TD', AddNoise, learn_snr_fake, learnAWGN)

                # Print
                if j == criticIter - 1 and iteration % plot_ite == 0:
                    plotfakereal(fakeamp=d_fake[:, 0].cpu().data.numpy(),
                                 realamp=d_real[:, 0].cpu().data.numpy(),
                                 ite=iteration, cite=j + 1, SaveFigPath=gan_result)

                # Real data is fed into the discriminator to obtain the model
                d_real = d_real.permute(1, 0, 2)
                d_real = d_real.unsqueeze(1)
                disc_real = netD(d_real)
                disc_real = disc_real.mean()

                #Simulated data fed into the discriminator to obtain the model
                d_fake = d_fake.permute(1, 0, 2)
                d_fake = d_fake.unsqueeze(1)
                disc_fake = netD(d_fake)
                disc_fake = disc_fake.mean()

                # Cumulative score of real and simulated data
                DiscReal = np.append(DiscReal, disc_real.item())
                DiscFake = np.append(DiscFake, disc_fake.item())

                # Gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, d_real, d_fake,
                                                         batch_size=num_shots_per_batch,
                                                         channel=1, lamb=lamb,
                                                         device=device)
                #Calculate total score
                disc_cost = disc_fake - disc_real + gradient_penalty
                print('Epoch: %03d  Ite: %05d  DLoss: %f' % (epoch + 1, iteration, disc_cost.item()))

                w_dist = -(disc_fake - disc_real)
                DLoss = np.append(DLoss, disc_cost.item())
                WDist = np.append(WDist, w_dist.item())

                # Based on total score for backpropagation
                disc_cost.backward()

                # Gradient Clipping to Prevent Gradient Explosion
                torch.nn.utils.clip_grad_norm_(netD.parameters(),
                                               1e3)  # 1e3 for smoothed initial model; 1e6 for linear model

                # optimize
                optim_d.step()

                # scheduler for discriminator
                if gan_weight_decay > 0:
                    scheduler_d.step()

                # -----------------Visualization---------------#
                if j == criticIter - 1 and iteration % plot_ite == 0:
                    PlotDLoss(dloss=DLoss, wdist=WDist,
                              SaveFigPath=gan_result)

                    # ------------Model update--------------#
            for p in netD.parameters():
                p.requires_grad_(False)  # This time, freeze the discriminator

            for k in range(1):
                optim_g.zero_grad()
                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    optim_noi.zero_grad()
                #Generate fake data
                g_fake = PhySimulator(model,
                                      source_amplitudes_init, it, criticIter, j,
                                      'TG', AddNoise, learn_snr, learnAWGN)

                g_fake = g_fake.permute(1, 0, 2)
                g_fake = g_fake.unsqueeze(1)

                if fix_value_depth > 0:
                    fix_model_grad(fix_value_depth, model)

                # Calculator generator loss
                gen_cost = netD(g_fake)
                gen_cost = gen_cost.mean()
                gen_cost.backward(mone)
                gen_cost = - gen_cost
                print('Epoch: %03d  Ite: %05d  GLoss: %f' % (epoch + 1, iteration, gen_cost.item()))
                GLoss = np.append(GLoss, gen_cost.item())

                # Gradient Clipping: Trim when exceeding the threshold
                torch.nn.utils.clip_grad_value_(model,
                                                1e3)  # mar_smal: 1e1 smoothed initial model; 1e3 for linear model

                # optimize
                optim_g.step()

                # Adjusting the learning rate of the generator
                if gan_weight_decay > 0:
                    scheduler_g.step()

                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    # Clips gradient value of learn_snr if needed
                    torch.nn.utils.clip_grad_value_(learn_snr, 1e1)
                    # optimize
                    optim_noi.step()
                    if gan_weight_decay > 0:
                        scheduler_noi.step()

                # clip ensure the value is greater than 0
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

            # ---------------Visualization---------------------#
            if iteration % plot_ite == 0:
                # Output the real model and the current inversion results
                plotcomparison(gt=model_true.cpu().data.numpy(),
                               pre=model.cpu().data.numpy(),
                               ite=iteration, SaveFigPath=gan_result)
                #Output GLoss
                PlotGLoss(gloss=GLoss, SaveFigPath=gan_result)

                #Output SNR, RSNR, SSIM and ERROR
                PlotSNR(SNR=SNR, SaveFigPath=gan_result)
                PlotSSIM(SSIM=SSIM, SaveFigPath=gan_result)
                PlotERROR(ERROR=ERROR, SaveFigPath=gan_result)
                if AddNoise is True and noise_var is not None and learnAWGN is True:
                    print('learned snr:', learn_snr)

        if (epoch + 1) % savepoch == 0 or (epoch + 1) == gan_num_epochs:
            # -----------------Save the result-----------------#
            spio.savemat(gan_result + 'GANRec' + '.mat',
                         {'rec': model.cpu().data.numpy()})
            spio.savemat(gan_result + 'GANMetric' + '.mat',
                         {'SNR': SNR, 'SSIM': SSIM, 'ERROR': ERROR,
                          'DLoss': DLoss, 'WDist': WDist, 'GLoss': GLoss})

            # ----------------------Save model----------------------#####
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
    print('Training Total Duration{:.0f}m  {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
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
