import torch
import vaetorch
from tqdm import tqdm
from torchvision.utils import save_image


class AE:
    def __init__(self, network, rec_loss, optimizer, save_path=None, save_img_path=None):
        super(AE, self).__init__()
        self.net = network
        self.rec_loss = rec_loss
        self.path = save_path
        self.optimizer = optimizer
        self.save_img_path = save_img_path

    def train(self, train_loader, val_loader, n_epochs, early_stop=None):
        early_counter = 0
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss = 0.0
            # train step
            for batch_idx, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.optimizer.zero_grad()
                X = X.view(X.shape[0], -1).to(vaetorch.device)
                X_rec = self.net(X)
                loss = self.rec_loss(X, X_rec)
                train_loss += loss
                loss.backward()
                self.optimizer.step()

            # validation step
            val_loss = 0.0
            first = True
            for batch_idx, (X, y) in enumerate(val_loader):
                with torch.no_grad():
                    X = X.view(X.shape[0], -1).to(vaetorch.device)
                    X_rec = self.net(X)
                    if self.save_img_path is not None and first:
                        # save grid of images after the epoch
                        save_image(X_rec[:50].view(-1, 1, 28, 28).cpu(),
                                   self.save_img_path + "/epoch_" + str(epoch + 1) + "-rec.jpg",
                                   nrow=10)
                        # generate images
                        eps = torch.randn((50, self.net.dec_layers[0].in_features)).to(vaetorch.device)
                        gen_images = self.net.dec(eps)
                        save_image(gen_images.view(-1, 1, 28, 28).cpu(),
                                   self.save_img_path + "/epoch_" + str(epoch + 1) + "-gen.jpg",
                                   nrow=10)
                        first = False
                    loss = self.rec_loss(X, X_rec)
                    val_loss += loss
            if early_stop is not None:
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    early_counter = 0
                    best_val_loss = val_loss
                    if self.path is not None:
                        self.save_model()
                else:
                    early_counter += 1
                    if early_counter >= early_stop:
                        if self.path is not None:
                            self.load_model()
                        print("Training interrupted due to early stopping")
                        break
            print("Epoch: %d - Train rec loss: %.3f - Val rec loss: %.3f" % (epoch + 1, train_loss / len(train_loader),
                                                                         val_loss))

    def predict(self, x):
        x = x.view(x.shape[0], -1).to(vaetorch.device)
        return self.net(x)

    def get_latent(self, x):
        x = x.view(x.shape[0], -1).to(vaetorch.device)
        return self.net.enc(x)

    def test(self, test_loader):
        test_loss = 0.0
        for batch_idx, (X, y) in enumerate(test_loader):
            with torch.no_grad():
                X = X.view(X.shape[0], -1).to(vaetorch.device)
                X_rec = self.net(X)
                loss = self.rec_loss(X, X_rec)
                test_loss += loss
        return test_loss / len(test_loader)

    def save_model(self):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.path)

    def load_model(self):
        checkpoint = torch.load(self.path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class VAE(AE):
    def __init__(self, network, rec_loss, kl_loss, optimizer, save_path=None, flat_input=True, save_img_path=None):
        super(VAE, self).__init__(network, rec_loss, optimizer, save_path, save_img_path)
        self.kl_loss = kl_loss
        self.flat_input = flat_input

    def train(self, train_loader, val_loader, n_epochs, early_stop=None):
        self.net.train()

        early_counter = 0
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss = 0.0
            train_kl_loss = 0.0
            train_rec_loss = 0.0
            # train step
            for batch_idx, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.optimizer.zero_grad()
                X = X.to(vaetorch.device)
                if self.flat_input:
                    X = X.view(X.shape[0], -1)
                mu, log_var, X_rec = self.net(X)
                rec_loss = self.rec_loss(X, X_rec)
                kl_loss = self.kl_loss(mu, log_var)
                loss = kl_loss + rec_loss
                train_loss += loss
                train_rec_loss += rec_loss
                train_kl_loss += kl_loss
                loss.backward()
                self.optimizer.step()

            # validation step
            val_loss = 0.0
            first = True
            for batch_idx, (X, y) in enumerate(val_loader):
                with torch.no_grad():
                    X = X.to(vaetorch.device)
                    if self.flat_input:
                        X = X.view(X.shape[0], -1)
                    mu, log_var, X_rec = self.net(X)
                    if self.save_img_path is not None and first:
                        # save grid of images after the epoch
                        save_image(X_rec[:50].cpu() if not self.flat_input else X_rec[:50].view(-1, 1, 28, 28).cpu(),
                                   self.save_img_path + "/epoch_" + str(epoch + 1) + "-rec.jpg",
                                   nrow=10)
                        # generate images
                        eps = torch.randn((50, self.net.latent_size)).to(vaetorch.device)
                        gen_images = self.net.dec(eps)
                        save_image(gen_images.cpu() if not self.flat_input else gen_images.view(-1, 1, 28, 28).cpu(),
                                   self.save_img_path + "/epoch_" + str(epoch + 1) + "-gen.jpg",
                                   nrow=10)
                        first = False
                    rec_loss = self.rec_loss(X, X_rec)
                    kl_loss = self.kl_loss(mu, log_var)
                    loss = kl_loss + rec_loss
                    val_loss += loss
            if early_stop is not None:
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    early_counter = 0
                    best_val_loss = val_loss
                    if self.path is not None:
                        self.save_model()
                else:
                    early_counter += 1
                    if early_counter >= early_stop:
                        if self.path is not None:
                            self.load_model()
                        print("Training interrupted due to early stopping")
                        break
            print("Epoch: %d - Train loss: %.3f - Train rec loss: %.3f - "
                  "Train KL loss: %.3f - Val loss: %.3f" % (epoch + 1, train_loss / len(train_loader),
                                                            train_rec_loss / len(train_loader),
                                                            train_kl_loss / len(train_loader),
                                                            val_loss))

    def predict(self, x):
        x = x.to(vaetorch.device)
        if self.flat_input:
            x = x.view(x.shape[0], -1)
        _, _, rec_x = self.net(x)
        return rec_x

    def get_latent(self, x):
        x = x.to(vaetorch.device)
        if self.flat_input:
            x = x.view(x.shape[0], -1)
        mu, _ = self.net.enc(x)
        return mu

    def test(self, test_loader):
        test_loss = 0.0
        for batch_idx, (X, y) in enumerate(test_loader):
            with torch.no_grad():
                x = x.to(vaetorch.device)
                if self.flat_input:
                    X = X.view(X.shape[0], -1)
                mu, log_var, X_rec = self.net(X)
                rec_loss = self.rec_loss(X, X_rec)
                kl_loss = self.kl_loss(mu, log_var)
                loss = kl_loss + rec_loss
                test_loss += loss
        return test_loss / len(test_loader)
