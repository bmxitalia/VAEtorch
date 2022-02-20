import torch
import vaetorch
import math


class AEnet(torch.nn.Module):
    """
    Vanilla autoencoder network architecture. The latent dimension has a linear activation. The output layer has a
    sigmoid activation.

    :param hidden_layers_size: list of integers containing the number of units of the fully-connected layers which
    compose the autoencoder architecture. The last dimension on the list is the latent dimension.
    Only the size of the layers for the encoder have to be passed. The decoder will have a symmetric architecture.
    :param act_func: activation function used in the hidden layers of the network. By default, the tanh activation
    function is used.
    """
    def __init__(self, hidden_layers_size, act_func=torch.nn.Tanh()):
        super(AEnet, self).__init__()
        self.enc_layers = torch.nn.ModuleList([torch.nn.Linear(i, j)
                                               for i, j in zip(hidden_layers_size[:-1], hidden_layers_size[1:])])
        self.dec_layers = torch.nn.ModuleList([torch.nn.Linear(i, j)
                                               for i, j in zip(hidden_layers_size[:0:-1], hidden_layers_size[-2::-1])])
        self.hidden_act = act_func

    def enc(self, x):
        """
        Encoder network.
        :param x: input of the encoder
        :return: the latent representation for the given input
        """
        for l in self.enc_layers[:-1]:
            x = self.hidden_act(l(x))
        return self.enc_layers[-1](x)

    def dec(self, z):
        """
        Decoder network.
        :param z: input of the decoder
        :return: the reconstruction for the given latent representation
        """
        for l in self.dec_layers[:-1]:
            z = self.hidden_act(l(z))
        return torch.sigmoid(self.dec_layers[-1](z))

    def forward(self, X):
        """
        Autoencoder network.
        :param X: the input of the autoencoder
        :return: the reconstructed version of the input
        """
        z = self.enc(X)
        return self.dec(z)


class VAEnet(torch.nn.Module):
    """
    VAE network architecture. The latent dimension has a linear activation (mu and log_var).
    The output layer has a sigmoid activation.

    :param hidden_layers_size: list of integers containing the number of units of the fully-connected layers which
    compose the VAE architecture.
    Only the size of the layers for the encoder have to be passed. The decoder will have a symmetric architecture.
    :param latent_size: size of the latent representation.
    :param act_func: activation function used in the hidden layers of the network. By default, the tanh activation
    function is used.
    """
    def __init__(self, hidden_layers_size, latent_size, act_func=torch.nn.Tanh()):
        super(VAEnet, self).__init__()
        self.enc_layers = torch.nn.ModuleList([torch.nn.Linear(i, j)
                                               for i, j in zip(hidden_layers_size[:-1], hidden_layers_size[1:])])
        self.dec_layers = torch.nn.ModuleList([torch.nn.Linear(i, j)
                                               for i, j in zip(hidden_layers_size[:0:-1], hidden_layers_size[-2::-1])])
        self.enc_layers.append(torch.nn.Linear(hidden_layers_size[-1], latent_size * 2))
        self.dec_layers.insert(0, torch.nn.Linear(latent_size, hidden_layers_size[-1]))
        self.latent_size = latent_size
        self.hidden_act = act_func

    def enc(self, x):
        """
        Encoder network.
        :param x: input of the encoder
        :return: the latent representation for the given input. These are two outputs, namely the mean and the log var
        of a Gaussian distribution.
        """
        for l in self.enc_layers[:-1]:
            x = self.hidden_act(l(x))
        x = self.enc_layers[-1](x)
        mu = x[:, :self.latent_size]
        log_var = x[:, self.latent_size:]
        return mu, log_var

    def dec(self, z):
        """
        Decoder network.
        :param z: input of the decoder. This is a sample from the distribution learned by the probabilistic encoder.
        This sample is obtained by doing a reparameterization trick.
        :return: the reconstruction for the given latent representation.
        """
        for l in self.dec_layers[:-1]:
            z = self.hidden_act(l(z))
        return torch.sigmoid(self.dec_layers[-1](z))

    def forward(self, X):
        """
        Autoencoder network. It performs a reparameterization trick to sample the latent vector that has to be given
        to the decoder.
        :param X: the input of the VAE.
        :return: the reconstructed version of the input.
        """
        mu, log_var = self.enc(X)
        # rep trick
        eps = torch.randn_like(mu).to(vaetorch.device)
        std = torch.exp(log_var * 0.5)
        z = (std * eps) + mu
        return mu, log_var, self.dec(z)


class CNNVAEnet(torch.nn.Module):
    """
    CNN VAE network architecture. The latent dimension has a linear activation (mu and log_var).
    The output layer has a sigmoid activation.

    :param cnn_act_maps: list of integers containing the number of activation maps of the CNN layers which
    compose the CNN VAE architecture.
    Only the activation maps of the layers for the encoder have to be passed. The decoder will have a symmetric
    architecture.
    :param latent_size: size of the latent representation;
    :param input_size: size of input images;
    :param kernel_size: size of the kernels used in CNN layers. The same size is used across all CNN layers;
    :param stride: stride for applying the convolution on the CNN layers. The same stride is used across all CNN layers;
    :param padding: padding for applying the convolution on the CNN layers. The same padding is used across all CNN
    layers. This padding is used also as output padding for the CNN transpose layers;
    :param act_func: activation function used in the hidden layers of the network. By default, the leaky relu activation
    function is used.
    """
    def __init__(self, cnn_act_maps, latent_size, input_size, kernel_size=3, stride=2, padding=1,
        act_func=torch.nn.LeakyReLU()):
        super(CNNVAEnet, self).__init__()
        get_out_size = lambda input_size: math.floor((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        out_size = input_size
        for i in range(len(cnn_act_maps) - 1):
            out_size = get_out_size(out_size)
        self.cnn_enc_layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Conv2d(i, j, kernel_size, stride, padding)) for i, j in
             zip(cnn_act_maps[:-1], cnn_act_maps[1:])])
        self.cnn_dec_layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.ConvTranspose2d(i, j, kernel_size, stride, padding, padding)) for i, j in
             zip(cnn_act_maps[:0:-1], cnn_act_maps[-2::-1])])
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.unflatten = torch.nn.Unflatten(1, (cnn_act_maps[-1], out_size, out_size))
        self.fc_enc = torch.nn.Linear(cnn_act_maps[-1] * out_size ** 2, latent_size*2)
        self.fc_dec = torch.nn.Linear(latent_size, cnn_act_maps[-1] * out_size ** 2)
        self.hidden_act = act_func
        self.latent_size = latent_size

    def enc(self, x):
        """
        Encoder network.
        :param x: input of the encoder
        :return: the latent representation for the given input. These are two outputs, namely the mean and the log var
        of a Gaussian distribution.
        """
        for l in self.cnn_enc_layers:
            x = self.hidden_act(l(x))
        x = self.flatten(x)
        x = self.fc_enc(x)
        mu = x[:, :self.latent_size]
        log_var = x[:, self.latent_size:]
        return mu, log_var

    def dec(self, z):
        """
        Decoder network.
        :param z: input of the decoder. This is a sample from the distribution learned by the probabilistic encoder.
        This sample is obtained by doing a reparameterization trick.
        :return: the reconstruction for the given latent representation.
        """
        z = self.fc_dec(z)
        # reshape the features to prepare con cnn layers
        z = self.unflatten(z)
        # cnn decoder layers
        for layer in self.cnn_dec_layers[:-1]:
            z = self.hidden_act(layer(z))
        return torch.sigmoid(self.cnn_dec_layers[-1](z))

    def forward(self, X):
        """
        Autoencoder network. It performs a reparameterization trick to sample the latent vector that has to be given
        to the decoder.
        :param X: the input of the VAE.
        :return: the reconstructed version of the input.
        """
        mu, log_var = self.enc(X)
        # rep trick
        eps = torch.randn_like(mu).to(vaetorch.device)
        std = torch.exp(log_var * 0.5)
        z = (std * eps) + mu
        return mu, log_var, self.dec(z)