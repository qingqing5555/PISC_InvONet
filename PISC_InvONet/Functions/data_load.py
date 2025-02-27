import scipy.io as spio
from scipy.signal import iirfilter
from scipy.signal import sosfilt
from scipy.signal import zpk2sos
import numpy as np
import torch
import scipy
import warnings
from deepwave import scalar

def loadrcv(rcv_filepath):
    """
        Load the receiver amplitude
    """
    data_mat = spio.loadmat(rcv_filepath)
    receiver_amplitudes_true = torch.from_numpy(np.float32(data_mat[str('true')]))
    return receiver_amplitudes_true


def loadtruemodel(data_dir, vmodel_dim):
    """
        加载真实模型
        输入：路径  dim【nz，ny】 我这里是100，310
    """
    # ndarray:31000 原数据是ny nz排列
    model_true = (np.fromfile(data_dir, np.float32).reshape(vmodel_dim[1], vmodel_dim[0]))
    # 这里转置一下变成 nz ny 转成torch.tensor
    model_true = torch.Tensor(np.transpose(model_true, (1, 0)))
    return model_true


def load_init_model(init_model_path):
    """
        加载初始模型
    """
    model_mat = spio.loadmat(init_model_path)
    # 将生成的数组（array）转换为张量Tensor
    model_init = torch.from_numpy(np.float32(model_mat[str('initmodel')]))
    return model_init


def createInitialModel(model_true, gfsigma, lipar, fix_value_depth):
    """
        对真实模型做处理，构建初始模型 ('line','lineminmax','const','GS')
    """
    assert gfsigma in ['line', 'lineminmax', 'constant', 'GS']
    # gpu上的数组不能直接进行转换类型，要先.cpu，前面model_true牵扯梯度计算，不能.numpy，所以要先.detach
    model_true = model_true.cpu().detach().numpy()
    shape = model_true.shape
    # 是否冻结部分
    if fix_value_depth > 0:
        const_value = model_true[:fix_value_depth, :]

    if gfsigma == 'line':
        # nz ny 在起始点和结束点间生成等间隔的数值序列
        value = np.linspace(model_true[fix_value_depth, np.int32(shape[1] / 2)],
                            model_true[-1, np.int32(shape[1] / 2)] * lipar, num=shape[0] - fix_value_depth,
                            endpoint=True, dtype=float).reshape(-1, 1)
        value = value.repeat(shape[1], axis=1)
    elif gfsigma == 'lineminmax':
        # line increased initial model (different min/max value)
        value = np.linspace(model_true.min() * lipar,
                            model_true.max(), num=shape[0] - fix_value_depth,
                            endpoint=True, dtype=float).reshape(-1, 1)

        value = value.repeat(shape[1], axis=1)
    elif gfsigma == 'const':
        # constant initial model
        value = model_true[fix_value_depth, int(np.floor(shape[1] / 2))] * np.ones(shape[0] - fix_value_depth, shape[1])
    # using Gaussian smoothed function
    else:
        value = scipy.ndimage.gaussian_filter(model_true[fix_value_depth:, :], sigma=5)

    if fix_value_depth > 0:
        model_init = np.concatenate([const_value, value], axis=0)
    else:
        model_init = value

    model_init = torch.tensor(model_init)
    print('model size:', model_init.size())

    return model_init


def createSR(n_shots, n_sources_per_shot, n_receivers_per_shot, d_source, first_source, d_receiver, first_receiver,
             source_depth, receiver_depth, device):
    """
        包含 源 和 接收器 的数组
        Args:
            n_shots: shots 爆炸次数 30
            n_sources_per_shot: 1
            n_receivers_per_shot: 接收器数量 310
            d_source: 震源间隔 注意是网格间隔
            first_source:
            d_receiver: 接收器间隔
            first_receiver:
            source_depth: 震源深度 0
            receiver_depth:接收器深度 0
            device:
        return:
            x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions] 第几次shot，震源坐标，几维
            x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions] 第几次shot，接收器坐标，几维
    """
    source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                   dtype=torch.long, device=device)
    source_locations[..., 0] = source_depth
    source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +
                                 first_source)

    receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                     dtype=torch.float32, device=device)
    receiver_locations[..., 0] = receiver_depth
    receiver_locations[:, :, 1] = (
        (torch.arange(n_receivers_per_shot) * d_receiver +
         first_receiver)
        .repeat(n_shots, 1)
    )
    return source_locations, receiver_locations


def loadinitsource(init_source_file):
    """
        加载初始震源
    """
    source_mat = spio.loadmat(init_source_file)
    source_init = torch.from_numpy(np.float32(source_mat[str('initsource')]))
    source_true = torch.from_numpy(np.float32(source_mat[str('truesource')]))

    return source_init, source_true


def createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots):
    """
        Create true source amplitudes [nt, num_shots, num_sources_per_shot]
        Args:
            peak_freq : 震源频率
            peak_source_time: delay 峰值源时间
            nt:时长
            dt: 周期
            num_shots:爆炸次数
        return:
            source_amplitudes

    """
    # 每个shot中的source都有一个ricker子波
    source_amplitudes_true = np.tile(ricker(peak_freq, nt, dt, peak_source_time).reshape(-1, 1, 1),
                                     [1, num_shots, 1])

    return source_amplitudes_true


def createFilterSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots,
                          use_filter, filter_type,
                          freqmin, freqmax, corners, df):
    """
        Create source amplitudes with filter function
        Args:
            peak_freq : 震源频率
            peak_source_time: delay 峰值源时间
            nt:时长
            dt: 周期
            num_shots:爆炸次数
            use_filter:
            filter_type:highpass','lowpass','bandpass'
            freqmax:
            freqmin:
            corners:
            df:
        return:返回滤波后的震源
    """

    source_amplitudes = ricker(peak_freq, nt, dt, peak_source_time)
    if use_filter:
        filt_data = seismic_filter(data=source_amplitudes,
                                   filter_type=filter_type, freqmin=freqmin,
                                   freqmax=freqmax, df=df, corners=corners)
        filt_data = filt_data
    else:
        filt_data = source_amplitudes

    source_amplitudes_filt = np.tile(filt_data.reshape(-1, 1, 1), [1, num_shots, 1])
    return source_amplitudes_filt


def ricker(freq, length, dt, peak_time):
    """
    Args:
        freq: A float specifying the central frequency of the wavelet
        length: An integer specifying the number of time steps to use
        dt: A float specifying the time interval between time steps
        peak_time: A float specifying the time (in time units) at which the
                   peak amplitude of the wavelet occurs

    Returns:
        A 1D Numpy array of length 'length' containing a Ricker wavelet
    """
    t = (np.arange(length) * dt - peak_time).astype(np.float32)
    y = (1 - 2 * np.pi ** 2 * freq ** 2 * t ** 2) \
        * np.exp(-np.pi ** 2 * freq ** 2 * t ** 2)
    return y

def seismic_filter(data, filter_type, freqmin, freqmax, df, corners, zerophase=False, axis=-1):
    """
    create the fileter for removing the frequency component of seismic data
    """
    assert filter_type.lower() in ['bandpass', 'lowpass', 'highpass']

    if filter_type == 'bandpass':
        if freqmin and freqmax and df:
            filt_data = bandpass(data, freqmin, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'lowpass':
        if freqmax and df:
            filt_data = lowpass(data, freqmax, df, corners, zerophase, axis)
        else:
            raise ValueError
    if filter_type == 'highpass':
        if freqmin and df:
            filt_data = highpass(data, freqmin, df, corners, zerophase, axis)
        else:
            raise ValueError
    return filt_data


def bandpass(data, freqmin, freqmax, df, corners, zerophase, axis):
    """
    Butterworth-Bandpass Filter.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def lowpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Lowpass Filter.
    Filter data removing data over certain frequency ``freq`` using ``corners``
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def highpass(data, freq, df, corners, zerophase, axis):
    """
    Butterworth-Highpass Filter.
    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis)
        return sosfilt(sos, firstpass[::-1], axis)[::-1]
    else:
        return sosfilt(sos, data, axis)


def add_noise(receiver_amplitudes, noise_type, noise_var):
    """
        添加噪声
    """
    if noise_type == 'Gaussian':
        receiver_amplitudes_noise = AddAWGN(data=receiver_amplitudes.cpu().data,
                                            snr=noise_var)
    return receiver_amplitudes_noise


def ComputeSNR(rec, target):
    """
       计算SNR
    """
    if torch.is_tensor(rec):
        rec = rec.cpu().data.numpy()
        target = target.cpu().data.numpy()

    if len(rec.shape) != len(target.shape):
        raise Exception('Please reshape the Rec and Target to correct Dimension!!')

    snr = 0.0
    if len(rec.shape) == 3:
        for i in range(rec.shape[0]):
            rec_ind = rec[i, :, :].reshape(np.size(rec[i, :, :]))
            target_ind = target[i, :, :].reshape(np.size(rec_ind))
            s = 10 * np.log10(sum(target_ind ** 2) / sum((rec_ind - target_ind) ** 2))
            snr = snr + s
        snr = snr / rec.shape[0]
    elif len(rec.shape) == 2:
        rec = rec.reshape(np.size(rec))
        target = target.reshape(np.size(rec))
        snr = 10 * np.log10(sum(target ** 2) / sum((rec - target) ** 2))
    else:
        raise Exception('Please reshape the Rec to correct Dimension!!')
    return snr


def AddAWGN(data, snr):
    """
       添加Gaussian noise
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


def createdata(model, dx, dt, source_amplitudes, source_locations, receiver_locations,
               order, pml_width, freq):
    """
        生成地震数据
    """
    # 震源从2000，30，1 改成 30，1，2000
    source_amplitudes = source_amplitudes.permute(1, 2, 0)
    out = scalar(model, dx, dt, source_amplitudes=source_amplitudes, source_locations=source_locations,
                 receiver_locations=receiver_locations, accuracy=order, pml_width=pml_width, pml_freq=freq)
    # Plot  [30,310,2000]
    receiver_amplitudes = out[-1]
    # vmin, vmax = np.percentile(receiver_amplitudes[0, :].cpu().numpy(), [2, 98])
    # 从30，310，2000 改成 2000,30,310
    source_amplitudes = receiver_amplitudes.permute(2, 0, 1)
    return source_amplitudes

