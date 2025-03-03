import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, DIM=128):
        super(Generator, self).__init__()
        self.DIM = DIM

        preprocess = nn.Sequential(
            nn.Linear(128, 8 * 8 * 8 * DIM),
            nn.BatchNorm1d(8 * 8 * 8 * DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(8 * DIM, 4 * DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True)
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 4, stride=2, padding=1)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8 * self.DIM, 8, 8)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 1, 128, 128)


class Generator_64(nn.Module):
    def __init__(self, DIM=128):
        super(Generator_64, self).__init__()
        self.DIM = DIM

        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 8 * DIM),
            nn.BatchNorm1d(4 * 4 * 8 * DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(8 * DIM, 4 * DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(4 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True)
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 4, stride=2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 4, stride=2, padding=1)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 8 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 1, 64, 64)


class Discriminator(nn.Module):
    def __init__(self, n_augments, DIM=128):
        super(Discriminator, self).__init__()
        self.n_augments = n_augments
        self.DIM = DIM

        main_conv = nn.Sequential(
            nn.Conv2d(1, DIM // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(DIM // 2, DIM, 4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(DIM, 2 * DIM, 4, stride=2, padding=1),
            nn.LeakyReLU()

        )

        main_linear = nn.Sequential(
            nn.Linear(8 * 8 * 2 * DIM, 1024),
            nn.Linear(1024, 256),
        )

        self.main_conv = main_conv
        self.main_linear = main_linear
        self.linear2 = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

        self.linears_dag = []
        for i in range(self.n_augments):
            self.linears_dag.append(
                nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
            )
        self.linears_dag = nn.ModuleList(self.linears_dag)

    def forward(self, input):

        conv_output = self.main_conv(input)
        feature = conv_output.view(-1, 8 * 8 * 2 * self.DIM)

        linear_output = self.main_linear(feature)
        output = self.linear2(linear_output)

        outputs_dag = []
        for i in range(self.n_augments):
            outputs_dag.append(self.linears_dag[i](linear_output))

        return output, outputs_dag