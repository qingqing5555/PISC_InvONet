import numpy as np

# --------------------Forward simulation parameters-----------------------------
peak_freq = 7.  # Frequency
peak_source_time = 1 / peak_freq  # Period
dx = 30.0  # Grid spacing
dt = 0.003  # Time interval 0.006 0.003
nz = 100  # z grid points mar_small:100  over:94  mar_big:117  openfwi:70
ny = 310  # y grid mar_small:310   over: 400   mar_big:567
vmodel_dim = np.array([nz, ny])  # Velocity model grid

total_t = 6  # Sampling time s
nt = int(total_t / dt)  # Total time sampling dimension
num_shots = 30  # shots times  mar_small:30 over:39 mar_big:56

num_sources_per_shot = 1  # Each shot has 1 source
d_source = 10  # Other: ny / (num_shots + 1) 310/31 * 30m = 300m  openfwi:400/40 *30=300m  10
first_source = 5  # Other: ny / (num_shots + 1) 310/31 * 30m = 300m  openfwi:10
source_depth = 0  # Source depth
source_spacing = np.floor(dx * ny / (num_shots + 1))  # Source spacing

num_receivers_per_shot = 310  # Receivers 310 400 567
d_receiver = 1  # 30m 1
first_receiver = 0  # 0m
receiver_depth = 0  # Receiver depth
receiver_spacing = np.floor(dx * ny / (num_receivers_per_shot + 1))  # Receiver spacing


survey_pad = None
fix_value_depth = 0  # Freeze parameters consistent with the real model

gfsigma = 'GS'  # Set different types to obtain different initial models
if gfsigma == 'line':
    lipar = 1.0
elif gfsigma == 'lineminmax':
    lipar = 1.1
else:
    lipar = None

order = 4  # Forward precision: generally low is 4, high is 8, can also be set higher
pml_width = [20, 20, 0, 20]   # Boundary thickness [Start of the first dimension, End of the first dimension, Start of the second dimension, End of the second dimension]

AddNoise = True  # Add Noise
if AddNoise == True:
    noise_type = 'Gaussian'
    noise_var = 20


# Filter (source) allows only a certain frequency range to pass through
use_filter = False  # Toggle
filter_type = 'highpass'  # Filter type
freqmin = 3  # Minimum frequency
freqmax = None  # Maximum frequency
corners = 6  # corners
df = 1 / dt

learnAWGN = False  # Learning noise level

# --------------------Traditional Inversion Parameters-----------------------------
fwi_lr = 2  # learning rate
fwi_batch = 30  # batches, not in deep learning, it is in the traditional deepwave inversion
fwi_num_epochs = 100
fwi_weight_decay = 0  # Adjust the parameter for lr, such as 0.1 (usually set to decrease lr gradually through gamma and step size)
if fwi_weight_decay > 0:
    fwi_stepsize = 100

AddTV = False  # ATV regularization item to prevent overfitting
if AddTV:
    alpha_tv = 1. * 1e-4  # weight for ATV

fscale = 1e2  # Multiply the scalar of normalized data to facilitate better backpropagation (can also be achieved by changing the learning rate)
data_norm = False  # normalizing
fwi_loss_type = 'L2'  # loss
savepoch = 50  # Save every 50 epochs
plot_ite = 100  # Output an image every 300 iterations

# --------------------FWIGAN parameters -----------------------------
gan_v_lr = 3 * 1e1  # Step size for updating the speed model
gan_num_batches = 6  # Insufficient memory for sampling, sample every 6 intervals, 30 shots can be divided into 6 blocks, each block with 5 shots (divided into many blocks) 6*5  Similarly, you can set the number of shots according to the size of your model
start_epoch = 0
gan_num_epochs = 300
DFilter = 32  # Input channels
leak_value = 0  # Negative slope of the activation function
gan_weight_decay = 0  # The weight_decay parameter can be set to control the decay of parameters during training, which is equivalent to the effect of L2 regularization
if gan_weight_decay > 0:
    gan_stepsize = 100  # Change the lr every 100 steps

gan_d_lr = 1 * 1e-3  # Discriminator learning rate

criticIter = 6  # Must be divisible by shots 6;3
lamb = 10  # Gradient parameter
