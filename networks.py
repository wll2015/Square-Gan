from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np


class VAEGen(nn.Module):

    def __init__(
            self, input_dim,
            basis_hid_layer_dims,
            trait_hid_layer_dims, decoder_hid_layer_dims, trait_dim):
        super(VAEGen, self).__init__()

        self.basis_encoder = Encoder(
            input_dim, basis_hid_layer_dims, trait_dim)
        self.trait_encoder = Encoder(
            input_dim, trait_hid_layer_dims, trait_dim)
        self.decoder = Decoder(
            trait_dim * 2,
            decoder_hid_layer_dims, input_dim)

    def encode(self, data):
        trait_fake = self.trait_encoder(data)
        basis = self.basis_encoder(data)
        return basis, trait_fake

    def forward(self, data):
        basis, trait_fake = self.encode(data)
        data_recon = self.decoder(torch.cat((basis, trait_fake), 1))
        return data_recon

    def decode(self, basis, trait):
        data = self.decoder(torch.cat((basis, trait), 1))
        return data


class Discriminator(nn.Module):

    def __init__(self, input_dim, hid_layer_dims, output_dim):
        super(Discriminator, self).__init__()

        _i = 0
        setattr(self, f'fc{_i}', nn.Linear(input_dim, hid_layer_dims[0]))
        _i += 1
        for ix, _dim in enumerate(hid_layer_dims[1:]):
            setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[ix], _dim))
            _i += 1
        setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[-1], 1))
        self.nlayers = _i

    def forward(self, x):
        out = None

        for _i in range(self.nlayers + 1):
            _fc = getattr(self, f'fc{_i}')
            if out is None:
                out = _fc(x)
            else:
                out = _fc(out)
            if _i < self.nlayers:
                out = F.relu(out)
            # else:
            #     out = F.sigmoid(out)

        return out

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random(
            (real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples
                                    )).requires_grad_(True)
        validity = self(interpolates)
        fake = Variable(torch.FloatTensor(np.ones(validity.shape)),
                        requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=validity, inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * 10

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        # outs0 = self.forward(input_fake)
        # outs1 = self.forward(input_real)
        loss = 0

        loss += self.compute_gradient_penalty(
                input_real, input_fake
            )

        # for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            # all0 = Variable(torch.zeros_like(out0.data), requires_grad=False)
            # all1 = Variable(torch.ones_like(out1.data), requires_grad=False)
            # loss += torch.mean(F.binary_cross_entropy(out0, all0) +
            #                    F.binary_cross_entropy(out1, all1))
            # loss += torch.mean(
            #     (out0 - 0)**2
            # ) + torch.mean((out1 - 1)**2) + self.compute_gradient_penalty(
            #     input_real, input_fake
            # )
            # loss += self.compute_gradient_penalty(
            #     input_real, input_fake
            # )
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            # all1 = Variable(torch.ones_like(out0.data), requires_grad=False)
            # loss += torch.mean(F.binary_cross_entropy(out0, all1))
            loss += torch.mean((out0 - 1)**2)
        return loss


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_layer_dims, output_dim):
        super(Encoder, self).__init__()

        _i = 0
        setattr(self, f'fc{_i}', nn.Linear(input_dim, hid_layer_dims[0]))
        _i += 1
        for ix, _dim in enumerate(hid_layer_dims[1:]):
            setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[ix], _dim))
            _i += 1
        setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[-1], output_dim))
        self.nlayers = _i
        # self.fcs.append(nn.Linear(input_dim, hid_layer_dims[0]))

        # for ix, _dim in enumerate(hid_layer_dims[1:]):
        #     self.fcs.append(nn.Linear(hid_layer_dims[ix], _dim))
        # self.fcs.append(nn.Linear(hid_layer_dims[-1], output_dim))
        self.activation = F.tanh

    def forward(self, x):
        out = None

        for _i in range(self.nlayers + 1):
            _fc = getattr(self, f'fc{_i}')
            if out is None:
                out = _fc(x)
            else:
                out = _fc(out)
            if _i < self.nlayers:
                out = F.leaky_relu(out)
            else:
                out = F.sigmoid(out)

        return out


class TraitEncoder(nn.Module):

    def __init__(self, input_dim):
        super(TraitEncoder, self).__init__()


class Decoder(nn.Module):

    def __init__(self, input_dim, hid_layer_dims, output_dim):
        super(Decoder, self).__init__()

        _i = 0
        setattr(self, f'fc{_i}', nn.Linear(input_dim, hid_layer_dims[0]))
        _i += 1
        for ix, _dim in enumerate(hid_layer_dims[1:]):
            setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[ix], _dim))
            _i += 1
        setattr(self, f'fc{_i}', nn.Linear(hid_layer_dims[-1], output_dim))
        self.nlayers = _i
        # self.fcs.append(nn.Linear(input_dim, hid_layer_dims[0]))

        # for ix, _dim in enumerate(hid_layer_dims[1:]):
        #     self.fcs.append(nn.Linear(hid_layer_dims[ix], _dim))
        # self.fcs.append(nn.Linear(hid_layer_dims[-1], output_dim))
        self.activation = F.tanh

    def forward(self, x):
        out = None

        for _i in range(self.nlayers + 1):
            _fc = getattr(self, f'fc{_i}')
            if out is None:
                out = _fc(x)
            else:
                out = _fc(out)
            if _i < self.nlayers:
                out = F.relu(out)
            else:
                out = F.sigmoid(out)

        return out
