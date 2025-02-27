import deepwave
import torch.nn as nn
from Functions.my_math import AddAWGN
from Functions.data_load import createdata


class PhySimulator(nn.Module):
    def __init__(self, dx, num_shots, num_batches, x_s, x_r, dt, pml_width, order, peak_freq):
        super(PhySimulator, self).__init__()
        self.dx = dx
        self.num_shots = num_shots
        self.num_batches = num_batches
        self.x_s = x_s
        self.x_r = x_r
        self.dt = dt
        self.num_shots_per_batch = int(self.num_shots / self.num_batches)
        self.pml_width = pml_width
        self.order = order
        self.peak_freq = peak_freq

    def forward(self, model, source_amplitudes, it, criticIter, j, status, 
                AddNoise, noi_var, learnAWGN):
        # 2000,5,1    5个shots  ;2000,13,1
        batch_src_amps = source_amplitudes.repeat(1, self.num_shots_per_batch, 1)
        if status == 'TD':

            # for the inner loop of training Critic
            test1 = it * criticIter + j
            if it * criticIter + j < self.num_batches:
                batch_x_s = self.x_s[it * criticIter + j::self.num_batches]  # 5,1,2  | 10,70,130,190,250
                batch_x_r = self.x_r[it * criticIter + j::self.num_batches]  # 5,310,2  | 0 ... 309
            else:
                batch_x_s = self.x_s[((it * criticIter + j) % self.num_batches)::self.num_batches]
                batch_x_r = self.x_r[((it * criticIter + j) % self.num_batches)::self.num_batches]

        elif status == 'TG':
            batch_x_s = self.x_s[it::self.num_batches]
            batch_x_r = self.x_r[it::self.num_batches]
        else:
            assert False, 'Please check the status of training!!!'

        # suzy 23 12 27   [2000,5,310]
        batch_rcv_amps_pred = createdata(model, self.dx, self.dt, batch_src_amps, batch_x_s, batch_x_r,
                   self.order, self.pml_width, self.peak_freq)

        if AddNoise == True and noi_var != None and learnAWGN == True:
            batch_rcv_amps_pred = AddAWGN(batch_rcv_amps_pred, noi_var)

        return batch_rcv_amps_pred
