from networks import VAEGen, Discriminator
from utils import weights_init, get_scheduler, get_model_list
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        # auto-encoder for domain a
        self.trait_dim = hyperparameters['gen']['trait_dim']

        self.gen_a = VAEGen(
            hyperparameters['input_dim'],
            hyperparameters['basis_encoder_dims'],
            hyperparameters['trait_encoder_dims'],
            hyperparameters['decoder_dims'], self.trait_dim)
        # auto-encoder for domain b
        self.gen_b = VAEGen(
            hyperparameters['input_dim'],
            hyperparameters['basis_encoder_dims'],
            hyperparameters['trait_encoder_dims'],
            hyperparameters['decoder_dims'], self.trait_dim)
        # discriminator for domain a
        self.dis_a = Discriminator(
            hyperparameters['input_dim'], hyperparameters['dis_dims'], 1)
        # discriminator for domain b
        self.dis_b = Discriminator(
            hyperparameters['input_dim'], hyperparameters['dis_dims'], 1)

        # fix the noise used in sampling
        self.trait_a = torch.randn(8, self.trait_dim, 1, 1)
        self.trait_b = torch.randn(8, self.trait_dim, 1, 1)

        # Setup the optimizers
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + \
            list(self.gen_b.parameters())
        for _p in gen_params:
            print(_p.data.shape)
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr, weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr, weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.gen_a.apply(weights_init('gaussian'))
        self.gen_b.apply(weights_init('gaussian'))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        trait_a = Variable(self.trait_a)
        trait_b = Variable(self.trait_b)
        basis_a, trait_a_fake = self.gen_a.encode(x_a)
        basis_b, trait_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(basis_b, trait_a)
        x_ab = self.gen_b.decode(basis_a, trait_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        trait_a = Variable(torch.randn(x_a.size(0), self.trait_dim))
        trait_b = Variable(torch.randn(x_b.size(0), self.trait_dim))
        # encode
        basis_a, trait_a_prime = self.gen_a.encode(x_a)
        basis_b, trait_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(basis_a, trait_a_prime)
        x_b_recon = self.gen_b.decode(basis_b, trait_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(basis_b, trait_a)
        x_ab = self.gen_b.decode(basis_a, trait_b)
        # encode again
        basis_b_recon, trait_a_recon = self.gen_a.encode(x_ba)
        basis_a_recon, trait_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(
            basis_a_recon, trait_a_prime
        ) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(
            basis_b_recon, trait_b_prime
        ) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_trait_a = self.recon_criterion(
            trait_a_recon, trait_a)
        self.loss_gen_recon_trait_b = self.recon_criterion(
            trait_b_recon, trait_b)
        self.loss_gen_recon_basis_a = self.recon_criterion(
            basis_a_recon, basis_a)
        self.loss_gen_recon_basis_b = self.recon_criterion(
            basis_b_recon, basis_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(
            x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(
            x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # total loss
        self.loss_gen_total = hyperparameters[
            'gan_w'] * self.loss_gen_adv_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
            hyperparameters['recon_trait_w'] * self.loss_gen_recon_trait_a + \
            hyperparameters['recon_basis_w'] * self.loss_gen_recon_basis_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_trait_w'] * self.loss_gen_recon_trait_b + \
            hyperparameters['recon_basis_w'] * self.loss_gen_recon_basis_b + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    # def sample(self, x_a, x_b):
    #     self.eval()
    #     s_a1 = Variable(self.s_a)
    #     s_b1 = Variable(self.s_b)
    #     s_a2 = Variable(torch.randn(x_a.size(0), self.trait_dim, 1, 1))
    #     s_b2 = Variable(torch.randn(x_b.size(0), self.trait_dim, 1, 1))
    #     x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
    #     for i in range(x_a.size(0)):
    #         c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
    #         c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
    #         x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
    #         x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
    #         x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
    #         x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
    #         x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
    #         x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
    #     x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
    #     x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
    #     x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
    #     self.train()
    #     return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        trait_a = Variable(torch.randn(x_a.size(0), self.trait_dim))
        trait_b = Variable(torch.randn(x_b.size(0), self.trait_dim))
        # encode
        basis_a, _ = self.gen_a.encode(x_a)
        basis_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(basis_b, trait_a)
        x_ab = self.gen_b.decode(basis_a, trait_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba, x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab, x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * \
            self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(),
                    'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(),
                    'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)
