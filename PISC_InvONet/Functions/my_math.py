import torch
from torch import autograd
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import numpy as np

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

def shot_max_normalize(x):
    """
   normalized to [-1, 1]
Channel is 1
    """
    num_shots, channel, num_time_steps, num_receivers = x.shape
    x_max, _ = torch.max(x.detach().reshape(num_shots, channel, num_time_steps * num_receivers).abs(), dim=2,
                         keepdim=True)
    x = x / (
        x_max.repeat(1, 1, num_time_steps * num_receivers).reshape(num_shots, channel, num_time_steps, num_receivers))
    return x


def Wasserstein1(f, g, trans_type, theta):
    """
        w1 loss
    """
    assert f.shape == g.shape
    assert len(f.shape) == 3
    device = f.device
    p = 1
    num_time_steps, num_shots_per_batch, num_receivers_per_shot = f.shape
    f = f.reshape(num_time_steps, num_shots_per_batch * num_receivers_per_shot)
    g = g.reshape(num_time_steps, num_shots_per_batch * num_receivers_per_shot)

    mu, nu, d = transform(f, g, trans_type, theta)

    assert mu.min() > 0
    assert nu.min() > 0

    mu = trace_sum_normalize(mu)
    nu = trace_sum_normalize(nu)

    F = torch.cumsum(mu, dim=0)
    G = torch.cumsum(nu, dim=0)

    w1loss = (torch.abs(F - G) ** p).sum()
    return w1loss


def transform(f, g, trans_type, theta):
    """
        Args:
           f, g: Seismic data shape: [Time, Explosion Times]
            trans_type: # linear, square, exp, softplus, abs
            theta: parameter
        """
    assert len(f.shape) == 2
    c = 0.0
    device = f.device
    if trans_type == 'linear':
        min_value = torch.min(f.detach().min(), g.detach().min())
        mu, nu = f, g
        c = -min_value if min_value < 0 else 0
        c = c * theta
        d = torch.ones(f.shape).to(device)
    elif trans_type == 'abs':
        mu, nu = torch.abs(f), torch.abs(g)
        d = torch.sign(f).to(device)
    elif trans_type == 'square':
        mu = f * f
        nu = g * g
        d = 2 * f
    elif trans_type == 'exp':
        mu = torch.exp(theta * f)
        nu = torch.exp(theta * g)
        d = theta * mu
    elif trans_type == 'softplus':
        mu = torch.log(torch.exp(theta * f) + 1)
        nu = torch.log(torch.exp(theta * g) + 1)
        d = theta / torch.exp(-theta * f)
    else:
        mu, nu = f, g
        d = torch.ones(f.shape).to(device)
    mu = mu + c + 1e-18
    nu = nu + c + 1e-18
    return mu, nu, d


def trace_sum_normalize(x):
    """
   Sum of traces and normalization
Channel is 1
    """
    x = x / (x.sum(dim=0, keepdim=True) + 1e-18)
    return x


def fix_model_grad(fix_value_depth, model):
    assert fix_value_depth > 0
    device = model.device
    # Gradient mask
    gradient_mask = torch.zeros(model.shape).to(device)
    # Depth below 1, above 0 [receiver_depth:,:] = 1; [:receiver_depth,:] = 0;
    gradient_mask[fix_value_depth:, :] = 1.0
    # Only update part 1 [receiver_depth:,:]
    model.register_hook(lambda grad: grad.mul_(gradient_mask))


def gaussian(window_size, sigma):
    """
    gaussian filter
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def ComputeSSIM(img1, img2, window_size=11, size_average=True):
    img1 = Variable(torch.from_numpy(img1))
    img2 = Variable(torch.from_numpy(img2))

    if len(img1.size()) == 2:
        d = img1.size()
        img1 = img1.view(1, 1, d[0], d[1])
        img2 = img2.view(1, 1, d[0], d[1])
    elif len(img1.size()) == 3:
        d = img1.size()
        img1 = img1.view(d[2], 1, d[0], d[1])
        img2 = img2.view(d[2], 1, d[0], d[1])
    else:
        raise Exception('The shape of image is wrong!!!')
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def create_window(window_size, channel):
    """
    create the window for computing the SSIM
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ComputeRE(rec, target):
    """
    Compute relative error between the rec and target
    """
    if torch.is_tensor(rec):
        rec = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()

    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')

    rec = rec.reshape(np.size(rec))
    target = target.reshape(np.size(rec))
    rerror = np.sqrt(sum((target - rec) ** 2)) / np.sqrt(sum(target ** 2))

    return rerror


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    L = 255
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def AddAWGN(data, snr):
    """
       Add additive white Gaussian noise to data such that the SNR is snr
    """
    if len(data.size()) != 3:
        assert False, 'Please check the data shape!!!'

    # change the shape to [num_shots,nt,num_receiver]
    data1 = data.permute(1, 0, 2)
    dim = data1.size()
    device = data1.device
    SNR = snr
    y_noisy = data1 + torch.randn(dim).to(device) * (
        torch.sqrt(torch.mean((data1.detach() ** 2).reshape(dim[0], -1), dim=1) / (10 ** (SNR / 10)))).reshape(dim[0],
                                                                                                               1,
                                                                                                               1).repeat(
        1, dim[1], dim[2])

    # change the shape to [nt,num_shots,num_receiver]
    y_noisy = y_noisy.permute(1, 0, 2)

    # check the shape of y_noisy is equal to data or not
    if y_noisy.size() != data.size():
        assert False, 'Wrong shape of noisy data!!!'

    return y_noisy

def createlearnSNR(init_snr_guess, device):
    """
        create learned snr when amplitude is noisy and try to learn the noise
    """
    learn_snr_init = torch.tensor(init_snr_guess)
    learn_snr = learn_snr_init.clone()
    learn_snr = learn_snr.to(device)
    # set_trace()
    learn_snr.requires_grad = True

    return learn_snr, learn_snr_init


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, channel, lamb, device):
    """
        Training the gradient penalty term for the discriminator (refer to WGAN_GP)
    """
    if batch_size != real_data.shape[0] or channel != real_data.shape[1]:
        assert False, "The batch size or channel is wrong!!!"
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    dim = real_data.size()
    alpha = alpha.view(batch_size, channel, dim[2], dim[3])
    alpha = alpha.float().to(device)

    fake_data = fake_data.view(batch_size, channel, dim[2], dim[3])
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return gradient_penalty
