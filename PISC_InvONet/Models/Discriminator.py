import torch.nn as nn
import torch.nn.init as init


# (6ConvBlock(Conv2d+LeakyReLU+MaxPool)+2FullyConnectedLayer)

def weights_init(m, leak_value):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:
            init.kaiming_normal_(m.weight, a=leak_value)

        if m.bias is not None:
            init.constant_(m.bias, 0.0)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.kaiming_normal_(m.weight, a=leak_value)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):
    def __init__(self, batch_size, ImagDim, LReLuRatio, filters, leak_value):
        super(Discriminator, self).__init__()

        self.truth_channels = 1
        self.batch_size = batch_size
        self.filters = filters
        self.LReLuRatio = LReLuRatio
        self.ImagDim = ImagDim
        self.leak_value = leak_value

        self.conv1 = nn.Conv2d(self.truth_channels, self.filters[0], kernel_size=3, stride=1, padding=1)
        self.ac1 = nn.LeakyReLU(self.LReLuRatio)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.filters[0], self.filters[1], kernel_size=3, stride=1, padding=1)
        self.ac2 = nn.LeakyReLU(self.LReLuRatio)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(self.filters[1], self.filters[2], kernel_size=3, stride=1, padding=1)
        self.ac3 = nn.LeakyReLU(self.LReLuRatio)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(self.filters[2], self.filters[3], kernel_size=3, stride=1, padding=1)
        self.ac4 = nn.LeakyReLU(self.LReLuRatio)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(self.filters[3], self.filters[4], kernel_size=3, stride=1, padding=1)
        self.ac5 = nn.LeakyReLU(self.LReLuRatio)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(self.filters[4], self.filters[5], kernel_size=3, stride=1, padding=1)
        self.ac6 = nn.LeakyReLU(self.LReLuRatio)
        self.pool6 = nn.MaxPool2d(2, 2)
        # mar_big: [2000,567]==>[31*8] (fc1:1000);
        # mar_smal:[2000,310]==>[31*4](fc1:2000) here now;
        # over:[2000,400]==>[31,6](fc1:1500)
        self.fc1 = nn.Linear(31 * 4 * filters[5], 2000)   # ori  31,8,2000  => 31*4*1024,2000
        self.ac7 = nn.LeakyReLU(self.LReLuRatio)
        self.fc2 = nn.Linear(2000, 1)  # ori 1500,1 =>2000,1

    def forward(self, input):
        # input shape: [num_shots_per_batch,1,nt,num_receiver_per_shot] [5,1,2000,310] ; [13 1 2000 310]
        output = input.reshape(self.batch_size, self.truth_channels, self.ImagDim[0], self.ImagDim[1])
        output = self.ac1(self.pool1(self.conv1(output)))  # in 5 1 2000 310 ;out 5 32 1000 155  13,32,1000,2000
        output = self.ac2(self.pool2(self.conv2(output)))  # in 5 32 1000 155 ; out 5 64 500 77  13 64 500 1000
        output = self.ac3(self.pool3(self.conv3(output)))  # in 5 64 500 77 ; out 5 128 250 38  13 128 250 50
        output = self.ac4(self.pool4(self.conv4(output)))  # in 5 128 250 38 ; out 5 256 125 19 13 256 125 25
        output = self.ac5(self.pool5(self.conv5(output)))  # in 5 256 125 19 ; out 5 512 62 9  13 512 62 12  8,512,62,17
        output = self.ac6(self.pool6(self.conv6(output)))  # in 5 512 62 9 ; out 5 1024 31 4   13,1024,31,6  8,1024,31,8
        # here the last dim should be same as fc1  ori:31 * 8 * 1024
        output = output.view(-1, 31 * 4 * 1024)  # out:5,126976(31*4*1024)   mar_small:4 over:6 不同剖面改一下参数
        output = self.fc1(output)  # out:5 2000 ;13 2000
        output = self.ac7(output)  # out:5 2000
        output = self.fc2(output)
        output = output.view(-1)
        return output

