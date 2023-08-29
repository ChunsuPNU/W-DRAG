import config as config

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.data import DataLoader
from util import *
from models import *


torch.manual_seed(13)

LAMBDA = 10


def calc_gradient_penalty(netD, real_data, fake_data, dag=False, dag_idx=0, batch_size=64, height=64):

    alpha = torch.rand(batch_size, 1).cuda()
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous().view(batch_size, 1, height, height)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if dag == False:
        disc_interpolates, _ = netD(interpolates)
    else:
        _, disc_interpolates = netD(interpolates)
        disc_interpolates = disc_interpolates[dag_idx]

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def D_loss_func(x_real, x_fake, netD, dag=False, dag_idx=0, batch_size=64, height=64):
    # real
    if dag == False:
        d_real, _ = netD(x_real)
        d_fake, _ = netD(x_fake)
    else:
        _, d_reals = netD(x_real)
        d_real = d_reals[dag_idx]
        _, d_fakes = netD(x_fake)
        d_fake = d_fakes[dag_idx]
    d_real = d_real.mean()
    d_fake = d_fake.mean()

    # train with gradient penalty

    gp = calc_gradient_penalty(netD, x_real, x_fake, dag=dag, dag_idx=dag_idx, batch_size=batch_size, height=height)
    # D cost
    d_cost = d_fake - d_real + gp
    return d_cost


def G_loss_func(x_real, x_fake, netD, dag=False, dag_idx=0):
    # fake
    if dag == False:
        d_fake, _ = netD(x_fake)
    else:
        _, d_fakes = netD(x_fake)
        d_fake = d_fakes[dag_idx]
    d_fake = d_fake.mean()
    # D cost
    g_cost = - d_fake
    return g_cost


def train(args):
    critic_iters = args.critic_iters  # For WGAN and WGAN-GP, number of critic iters per gen iter
    batch_size = args.batch_size
    DIM = args.dim
    policy = args.policy
    policy_weight = args.policy_weight
    num_epochs = args.num_epochs
    subset_folder = args.subset_dir
    class_mode = args.class_mode
    train_continue = args.train_continue
    date = args.date
    aug_dloss_weight = args.aug_dloss_weight
    aug_gloss_weight = args.aug_gloss_weight
    save_memo = args.save_memo
    height = args.height
    disc_model = args.disc_model
    n_optim = args.n_optim

    if len(policy) > 1:
        aug_method = ''.join([word[:4] for word in policy])
    else:
        aug_method = policy[0]

    print("aug method: %s" % aug_method)
    print("batch size: %d" % batch_size)
    print("policy: %s" % policy)
    print("policy weight: %s" % policy_weight)
    print("data folder (subset): %s" % subset_folder)
    print("class mode: %s" % class_mode)

    data_dir = f'./data/{subset_folder}/x_train_{class_mode}.npy'
    ckpt_dir = f'./{date}/ckpt/{subset_folder}/{aug_method}_{save_memo}/'

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    dag = DAG(D_loss_func, G_loss_func, policy=policy, policy_weight=policy_weight)
    n_augments = dag.get_num_of_augments_from_policy()

    if height == 64:
        netG = Generator_64(DIM=DIM).cuda()
    else:
        netG = Generator(DIM=DIM).cuda()

    netD = Discriminator(n_augments=n_augments, DIM=DIM).cuda()


    init_weights(netG, init_type='kaiming', init_gain=0.02)
    init_weights(netD, init_type='kaiming', init_gain=0.02)

    if n_optim == '0':
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


    st_epoch = 0
    if train_continue == 'on':
        netG, netD, optimizerG, optimizerD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD,
                                                            optimG=optimizerG, optimD=optimizerD)

    train_dataset = np.load(data_dir)[:, np.newaxis, :, :]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

    total_iters = 0

    for epoch in range(st_epoch + 1, num_epochs + 1):

        for batch, data in enumerate(train_loader, 1):
            total_iters += 1
            for p in netD.parameters():
                p.requires_grad = True

            real_data = torch.stack([preprocess(item) for item in data])
            real_data = real_data.cuda()
            real_data_v = autograd.Variable(real_data).float()

            # train with fake
            noise = torch.randn(batch_size, 128)
            noise = noise.cuda()

            with torch.no_grad():
                noisev = noise
            # fake = autograd.Variable(netG(noisev).data)
            fake = autograd.Variable(netG(noisev).data).float()
            inputv = fake


            # D_cost = D_loss_func(real_data_v, inputv, netD, batch_size=batch_size, height=height)
            D_cost = D_loss_func(real_data_v, inputv, netD, batch_size=batch_size, height=height) \
                     + aug_dloss_weight * dag.compute_discriminator_loss(real_data_v, inputv, netD)
            netD.zero_grad()
            D_cost.backward()
            optimizerD.step()


            if total_iters % critic_iters == 0:
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()

                noise = torch.randn(batch_size, 128)
                noise = noise.cuda()
                noisev = autograd.Variable(noise)
                # fake = autograd.Variable(netG(noise).data).float()
                fake = netG(noisev)

                G_cost = G_loss_func(None, fake, netD) + aug_gloss_weight * dag.compute_generator_loss(real_data_v, inputv, netD)
                # G_cost = G_loss_func(None, fake, netD)

                G_cost.backward()
                optimizerG.step()




