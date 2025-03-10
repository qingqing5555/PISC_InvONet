from torch import nn
import deepwave
from deepwave import scalar
import network
from vis import *
import utils
from pytorch_msssim import ms_ssim


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# --------------------------Set parameters-------------------------------------#
model_type = 'OPENFWI'
if model_type == 'SEG':
    vis_path = './OpenFWI/FWI/SEG/'

    num = 1

    models = np.load('../open_data/Simulatedata/model1_8.npy')
    datas = np.load('../OpenFWI/FWI/data/data60.npy')

    ny = 100  # [201 301] -> 100 300
    nx = 300  #
    dx = 30.0  #
    n_shots = 30  #
    n_sources_per_shot = 1
    d_source = 10  # 17.25 * 10m = 172.5m
    first_source = 5  # 0 * 10m = 0m
    source_depth = 0  # 0m

    n_receivers_per_shot = 150
    d_receiver = 2  # 1 * 10m = 10m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 0  # 0m

    freq = 8
    nt = 1000
    dt = 0.006
    peak_time = 1.5 / freq
    AddNoise = False
    noise_var = 0.03
    data_min = -29.04  # -26 float
    data_max = 57.03  # 51
    model_min = 1500
    model_max = 4500
else:
    # 如果是openfwi data
    num = 2  # 剖面编号
    plot_num = 20
    vis_path = './FWI/FWI_checkpoint/' + 'model' + str(num) + '/'
    models = np.load('./FWI/model/model60.npy')
    datas = np.load('./FWI/data/data60.npy')
    ny = 70  #
    nx = 70  #
    dx = 10.0  #
    n_shots = 5  #
    n_sources_per_shot = 1
    d_source = 17.25  # 17.25 * 10m = 172.5m
    first_source = 0  # 0 * 10m = 0m
    source_depth = 0  # 0m

    n_receivers_per_shot = 70
    d_receiver = 1  # 1 * 10m = 10m
    first_receiver = 0  # 0 * 4m = 0m
    receiver_depth = 0  # 0m

    freq = 15
    nt = 1000
    dt = 0.001
    peak_time = 1.5 / freq

    AddNoise = False
    noise_var = 0.03

    data_min = -29.04  # -26 float
    data_max = 57.03  # 51
    model_min = 1500
    model_max = 4500


# --------------------------Generate data---------------------------------------------------------#
def generate_data(data_name, save_path='./open_data/SGEdata/data_', accuracy=4):
    """
       This function can be used to generate your own seismic data. Currently, it defaults to processing 500 profiles at a time, with the models dimension set to [Length][Width][0], which can also be modified by yourself
        Args:
            data_name
            save_path
            accuracy: Accuracy 4 or 8
    """
    my_data = []
    for i in range(0, 500):
        # 1 Load real model data
        model_true = models[i][0]
        # model_true = model_true[0:-1, 0:-1]   # open:5*1000*70   SEG:30*1000*300

        # 2 Seismic Source
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                       dtype=torch.long, device=device)
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +  # 5 10...295
                                     first_source)

        # 3 Seismic data
        # Receiver
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                         dtype=torch.long, device=device)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = (
            (torch.arange(n_receivers_per_shot) * d_receiver +
             first_receiver)
            .repeat(n_shots, 1)
        )  # 0 1 ...299
        source_amplitudes = (
            (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
            .repeat(n_shots, n_sources_per_shot, 1).to(device)
        )

        # observed_data = torch.rand([n_shots, n_receivers_per_shot, nt]).to(device)  # 5 70 1000   30 300 1000
        observed_data = scalar(torch.tensor(model_true, dtype=torch.float32).to(device), dx, dt,
                               source_amplitudes=source_amplitudes.to(device),
                               source_locations=source_locations.to(device),
                               receiver_locations=receiver_locations.to(device),
                               accuracy=accuracy,
                               pml_freq=freq)
        observed_data = observed_data[-1] * (-1)  # 5 70 1000
        #  Add noise
        if AddNoise == True and noise_var != None:
            noise = np.random.normal(0, noise_var, observed_data[0].shape)
            data = torch.clip(observed_data[0] + torch.tensor(noise, dtype=torch.float32),
                              min=-22, max=41).numpy()
            # plot_single_seismic(observed_data[0].cpu().permute(1, 0).numpy(), 'F:/suzy/OpenFWI/FWI/data/1.jpg')

        # plot_single_seismic(observed_data[0].cpu().permute(1, 0).numpy(), 'F:/suzy/OpenFWI/FWI/data/1.jpg')
        # 5, 1000, 70   30 300 2000->30 2000 300
        my_data.append(torch.unsqueeze(observed_data.cpu().permute(0, 2, 1), 0))

    np.save(save_path + str(data_name), torch.cat(my_data))
    # np.save('F:/suzy/open_data/Simulatedata/data1_' + str(8), torch.cat(my_data))
    print('Data generation successful~')


# ------------------------normalize func----------------------#
def log_transform(data, k=1, c=0):
    # log
    log = (torch.log1p(torch.abs(k * data) + c)) * torch.sign(data)
    return log


def MinMaxNormalize(data, min, max, scale=2):
    data = data - min
    data = data / (max - min)  # 0-1
    return (data - 0.5) * 2 if scale == 2 else data  # -1 - 1


def log_minmaxNormalize(data, min, max, scale=2):
    return MinMaxNormalize(log_transform(data), log_transform(torch.tensor(min)), log_transform(torch.tensor(max)))


def exp_transform(data, k=1, c=0):
    return (torch.expm1(torch.abs(data)) - c) * torch.sign(data) / k


def minmax_denormalize(data, min, max, scale=2):
    if scale == 2:
        data = data / 2 + 0.5
    return data * (max - min) + min


def denormalize(data, min, max, exp=True, k=1, c=0, scale=2):
    if exp:
        min = log_transform(torch.tensor(min), k=k, c=c)
        max = log_transform(torch.tensor(max), k=k, c=c)
    data = minmax_denormalize(data, min, max, scale)
    return exp_transform(data, k=k, c=c) if exp else data


# ------------------------Iterative training----------------------------------#
def my_train(num=1, model_type='ConvModNet', lr=0.0001,
             checkpoint_path='./FWI/FWI_checkpoint/checkpoint.pth', train_epochs=100,
             plot_num=1):
    """
        Args:
             num: Profile number for inversion
             model_type：Network Model
             lr：Learning Rate
             checkpoint_path：Pre-trained model './FWI/FWI_checkpoint/model2/checkpoint_for_1.pth'
             train_epochs：Iteration Round
             plot_num：Output: Output the loss and images at regular epochs
    """
    if not os.path.exists(vis_path):
        utils.mkdir(vis_path)

    # Loading real seismic data and profiles
    first = int(num / 16) * 16
    last = first + 16
    invert_num = num % 16 - 1
    true_data = torch.from_numpy(datas[first:last, :]).to(device)  # 16 5 1000 70
    plot_single_seismic(true_data[invert_num][0].cpu().numpy(), vis_path + 'true_seis.jpg')

    true_model = torch.unsqueeze(torch.from_numpy(models[num - 1]).to(device), 0)  
    plot_single_velocity(true_model[0][0].cpu().numpy(), vis_path + 'true_model.jpg')

    data_min = torch.min(true_data)
    data_max = torch.max(true_data)
    model_min = torch.min(true_model)
    model_max = torch.max(true_model)
    true_data_norm = log_minmaxNormalize(true_data, data_min, data_max)  
    true_data_criterion = torch.unsqueeze(true_data_norm[invert_num, :], 0)  Real seismic data for comparison
    # Load model and parameters
    model = network.model_dict[model_type](upsample_mode=None,
                                           sample_spatial=1.0, sample_temporal=1, norm='bn').to(
        device)

    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def criterion(pred, gt, lambda_g1v=0.3, lambda_g2v=0.7):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss_ssim_l1 = 1 - ms_ssim(gt / 2 + 0.5, pred / 2 + 0.5, data_range=1, size_average=True, win_size=5)

        loss = lambda_g1v * loss_g1v + lambda_g2v * loss_ssim_l1
        return loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = network.load_dict_to_model(checkpoint['model'], model)

    # Forward modeling parameters
    # Seismic Source
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long, device=device)
    source_locations[..., 0] = source_depth
    source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +
                                 first_source)
    # Receiver
    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.long, device=device)
    receiver_locations[..., 0] = receiver_depth
    receiver_locations[:, :, 1] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
         first_receiver)
        .repeat(n_shots, 1)
    )
    source_amplitudes = (
        (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
        .repeat(n_shots, n_sources_per_shot, 1).to(device)
    )

    # Start training
    print('Start training')
    loss = 0.0
    best_loss = 100
    for epoch in range(0, train_epochs):
        model.eval()
        model.zero_grad()
        optimizer.zero_grad()

        # if epoch % 2 == 0:
        #     pre_model = model(true_data_norm[:, :, 0:1000:2, 0:300:2])  # 16 5 1000 300
        # else:
        #     pre_model = model(true_data_norm[:, :, 1:1000:2, 1:300:2])  # 16 5 1000 300
        pre_model = model(true_data_norm)  # 16 5 1000 300
        # Forward calculation, then calculate loss with true_data
        pre_model_denorm = denormalize(pre_model[invert_num], model_min, model_max, False)  # 1 100 300
        if epoch == 0:
            plot_velocity(pre_model_denorm[0].cpu().detach().numpy(), true_model[0][0].cpu().numpy(),
                          vis_path + 'pre_model.jpg')
        # plot_velocity(pre_model_denorm[0].cpu().detach().numpy(), true_model[0][0].cpu().numpy(), vis_path + 'pre_model.jpg')
        # -------------------------
        observed_data = scalar(pre_model_denorm[0], dx, dt,
                               source_amplitudes=source_amplitudes.to(device),
                               source_locations=source_locations.to(device),
                               receiver_locations=receiver_locations.to(device),
                               accuracy=4,
                               pml_freq=freq)
        observed_data = observed_data[-1] * (-1)  
        plot_single_seismic(observed_data[0].cpu().detach().permute(1, 0).numpy(), vis_path + 'pre_seis.jpg')
        pre_data_norm = log_minmaxNormalize(observed_data.permute(0, 2, 1), data_min, data_max)  # 5 1000 70
        # -------------------------
        loss = criterion(torch.unsqueeze(pre_data_norm, 0), true_data_criterion)  # 1 5 1000 70
        print("epoch:", epoch)
        print("loss:", loss.item())
        loss.backward()
        #Trimming to prevent gradient vanishing between backward and step
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        model.zero_grad()

        # Output: Output the loss and images at regular epochs
        if epoch % plot_num == 0 and epoch > 0:
            plot_velocity(pre_model_denorm[0].cpu().detach().numpy(), true_model[0][0].cpu().detach().numpy(),
                          f'{vis_path}/V_{plot_num}_{epoch}.png')
        # Save model
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(
                checkpoint,
                os.path.join(vis_path, 'checkpoint_for_' + str(num) + '.pth'))
            print('saving checkpoint at epoch: ', epoch)
            best_loss = loss


if __name__ == '__main__':
    # Generate data
    # generate_data(1)
    # Iterative Network Model
    my_train(num=num, model_type='ConvModNet')  # InversionNet ConvModNet

