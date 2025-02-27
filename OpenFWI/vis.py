import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.fft import fft, fftfreq

# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

def plot_velocity_seg(output, target, path, vmin=1471.777, vmax=5772.396):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap='jet', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.2)
    ax[1].matshow(target, cmap='jet', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.2)

    for axis in ax:
        if (output.shape[0] == 70):
            axis.set_xticks(range(0, 70, 10))
            axis.set_xticklabels(range(0, 700, 100), fontsize=11)
            axis.set_yticks(range(0, 70, 10))
            axis.set_yticklabels(range(0, 700, 100), fontsize=11)
        else:
            axis.set_xticks(range(0, 300, 50))
            axis.set_xticklabels(range(0, 9000, 1500), fontsize=11)
            axis.set_yticks(range(0, 100, 50))
            axis.set_yticklabels(range(0, 3000, 1500), fontsize=11)
            axis.yaxis.set_tick_params(rotation=90)
        axis.set_ylabel('Depth (m)', fontsize=11)
        axis.set_xlabel('Offset (m)', fontsize=11)
        axis.xaxis.set_ticks_position('top')
        plt.subplots_adjust(wspace=0.3)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

def plot_velocity(output, target, path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)

    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100), fontsize=11)
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100), fontsize=11)

        axis.set_ylabel('Depth (m)', fontsize=11)
        axis.set_xlabel('Offset (m)', fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

def plot_single_velocity(label, path):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    vmax, vmin = np.max(label), np.min(label)
    im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    # im = ax.matshow(label, cmap='jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=1.0, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')

# def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
#     fig, ax = plt.subplots(1, 3, figsize=(15, 6))
#     im = ax[0].matshow(output, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[0].set_title('Prediction')
#     ax[1].matshow(target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[1].set_title('Ground Truth')
#     ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
#     ax[2].set_title('Difference')
#     fig.colorbar(im, ax=ax, format='%.1e')
#     plt.savefig(path)
#     plt.close('all')


def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    aspect = output.shape[1]/output.shape[0]
    im = ax[0].matshow(target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[1].matshow(output, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Prediction')
    ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Difference')
    
    # for axis in ax:
    #     axis.set_xticks(range(0, 70, 10))
    #     axis.set_xticklabels(range(0, 1050, 150))
    #     axis.set_title('Offset (m)', y=1.1)
    #     axis.set_ylabel('Time (ms)', fontsize=12)
    
    # fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    fig.colorbar(im, ax=ax, shrink=0.75, label='Amplitude')
    plt.savefig(path)
    plt.close('all')


def plot_single_seismic(data, path):
    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap='gray', vmin=vmin * 0.01, vmax=vmax * 0.01)
    ax.set_aspect(aspect=nx/nz)
    ax.set_xticks(range(0, nx, int(300//(1050/nx)))[:5])
    ax.set_xticklabels(range(0, 1050, 300))
    ax.set_title('Offset (m)', y=1.08)
    ax.set_yticks(range(0, nz, int(200//(1000/nz)))[:5])
    ax.set_yticklabels(range(0, 1000, 200))
    ax.set_ylabel('Time (ms)', fontsize=18)
    fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    plt.savefig(path)
    plt.close('all')

def plotinitmodel(init, true, vmin, vmax, SaveFigPath):
    """
    plot initial velocity model
    """

    model_init = init
    model_true = true
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)
    plt.subplot(1, 2, 1)
    plt.imshow(model_init.cpu().detach().numpy(), vmin=vmin, vmax=vmax,
               cmap='jet')
    plt.colorbar()
    plt.title('inital model')
    plt.subplot(1, 2, 2)
    plt.imshow(model_true.cpu().detach().numpy(), vmin=vmin, vmax=vmax,
               cmap='jet')
    plt.colorbar()
    plt.title('ground truth')
    plt.savefig(SaveFigPath + 'init_model.png')
    plt.close()


def plotinitsource(init, gt, SaveFigPath):
    """
        打印震源
    """
    t = 500
    figsize = (12, 6)
    plt.figure(figsize=figsize)
    plt.plot(init.reshape(-1).ravel()[:t], label='Initial')
    plt.plot(gt.reshape(-1).ravel()[:t], label='True')
    plt.legend()

    plt.title('source amplitude')

    plt.savefig(SaveFigPath + 'source.png')

    plt.close()


def plotsourcespectra(init_source, true_source, SaveFigPath):
    """
        震源频谱图
    """
    init_source = init_source.astype(np.complex64).reshape(-1)
    true_source = true_source.astype(np.complex64).reshape(-1)
    true_yf = fft(true_source).reshape(-1)
    true_xf = fftfreq(len(true_source), 0.003)
    init_yf = fft(init_source).reshape(-1)

    idx = np.argsort(true_xf)
    xx = true_xf[idx]
    iyx = (np.abs(init_yf))[idx]
    tyx = (np.abs(true_yf))[idx]

    figsize = (6, 6)
    plt.figure(figsize=figsize)
    plt.plot(xx[len(xx) // 2:len(xx) // 2 + 200], iyx[len(xx) // 2:len(xx) // 2 + 200], label='Initial')
    plt.plot(xx[len(xx) // 2:len(xx) // 2 + 200], tyx[len(xx) // 2:len(xx) // 2 + 200], label='True')
    plt.legend()

    plt.title('source frequency spectra')

    plt.savefig(SaveFigPath + 'spectra.png')

    plt.close()


def plotoneshot(receiver_amplitudes_true, SaveFigPath):
    """
        one shot data 1000 5 70
    """
    fig1, ax1 = plt.subplots()
    # note numpy.percentile(a, q, axis=None, out=None, overwrite_input=False,   interpolation='linear', keepdims=False) compute q-th percentile(s)
    # compute the 2% and 98% of summation of array
    vmin, vmax = np.percentile(receiver_amplitudes_true[:, 0].cpu().numpy(), [2, 98])
    shot = 3
    # show = receiver_amplitudes_true[:, 0].cpu().numpy()
    plt.imshow(receiver_amplitudes_true[:, shot].cpu().numpy(), aspect='auto',
               vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar()
    plt.title('One shot data')

    plt.savefig(SaveFigPath + 'data_shot_' + str(shot) + '.png')

    plt.close()


def plotcomparison(gt, pre, ite, SaveFigPath):
    """
    打印真实模型和反演模型
    """

    dim = gt.shape
    gt = gt.reshape(dim[0], dim[1])
    pre = pre.reshape(dim[0], dim[1])

    vmin, vmax = np.percentile(gt, [2, 98])
    fig1, ax1 = plt.subplots(2)
    fig1.set_figheight(6)
    fig1.set_figwidth(12)

    plt.subplot(1, 2, 1)
    plt.imshow(pre, vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('inverted model')

    plt.subplot(1, 2, 2)
    plt.imshow(gt, vmin=vmin, vmax=vmax, cmap='jet')
    plt.colorbar()
    plt.title('true model')

    plt.savefig(SaveFigPath + 'invert_ite{}.png'.format(ite))

    plt.close()
