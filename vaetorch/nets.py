import torch
import vaetorch


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