# -*- encoding:utf-8 -*-
# author: liuheng

import torch
from torch import nn
from torch.nn import functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(BidirectionalLSTM, self).__init()

        self.lstm = nn.LSTM(in_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, out_size)

    def forward(self, inputs):
        out, h = self.lstm(inputs)
        batchsz, T, h = out.shape
        t_out = out.view(T * batchsz, h)

        output = self.fc(t_out)
        return output.view(T, batchsz, -1)

class CRNN(nn.Module):
    def __init__(self, imgH, n_channels, n_classes, n_hidden, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0,  'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        CNN_Seq = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            in_size = n_classes if i == 0 else nm[i - 1]
            out_size = nm[i]

            CNN_Seq.add_module('conv{0}'.format(i),
                               nn.Conv2d(in_size, out_size, kernel_size=ks[i], stride=ss[i], padding=ps[i]))
            if batchNormalization == True:
                CNN_Seq.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(out_size))
            if leakyRelu:
                CNN_Seq.add_module('leakyRelu{0}'.format(i),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                CNN_Seq.add_module('Relu{0}'.format(i), nn.ReLU(inplace=True))

        convRelu(0)
        CNN_Seq.add_module(f'pooling{0}', nn.MaxPool2d(kernel_size=2, stride=2))  # 64x16x64
        convRelu(1)
        CNN_Seq.add_module(f'pooling{1}', nn.MaxPool2d(kernel_size=2, stride=2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        CNN_Seq.add_module(f'pooling{2}',
                           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)

        CNN_Seq.add_module(f'pooling{3}',
                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = CNN_Seq
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, n_hidden, n_hidden),
            BidirectionalLSTM(n_hidden, n_hidden, out_size=n_classes)
        )

    def forward(self, inputs):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output