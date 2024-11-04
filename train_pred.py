import torch
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd


def reparameterize(mu, log):
    std = log.mul(0.5).exp_()
    esp = torch.randn(*mu.size())
    z = mu+std*esp.cuda()
    return z

def pred(test_loader, encoder, decoder, Disc, ae_criterion, args):

    device = args.device
    EPS = args.EPS
    test_loss = []
    test_rec_loss = 0
    test_disc_loss = 0
    test_gen_loss = 0
    test_labels = []

    encoder.load_state_dict(torch.load(args.model_path + 'encoder_epoch'+str(args.num_epoch)+'.pt'))
    decoder.load_state_dict(torch.load(args.model_path + 'decoder_epoch'+str(args.num_epoch)+'.pt'))
    Disc.load_state_dict(torch.load(args.model_path + 'disc_epoch'+str(args.num_epoch)+'.pt'))

    for i, (data, labels) in enumerate(test_loader):
        encoder.eval()
        decoder.eval()
        Disc.eval()

        with torch.no_grad():
            """ Reconstruction loss """
            for p in Disc.parameters():
                p.requires_grad = False

            real_data_v = autograd.Variable(data).to(device)
            test_labels.append(labels)
            enc_mu, enc_var = encoder(real_data_v)
            encoding = reparameterize(enc_mu, enc_var)
            fake = decoder(encoding)
            ae_loss = ae_criterion(fake, real_data_v)

            test_rec_loss += ae_loss.item()
            test_loss.append(ae_loss.item())

            """ Discriminator loss """
            encoder.eval()
            z_real_gauss = autograd.Variable(torch.randn(data.size()[0], 32, 22, 61) * 5.).to(device)
            D_real_gauss = Disc(z_real_gauss)

            mu_fake_gauss, var_fake_gauss = encoder(real_data_v)
            z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
            D_fake_gauss = Disc(z_fake_gauss)

            D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
            test_disc_loss += D_loss.item()

            mu_fake_gauss, var_fake_gauss = encoder(real_data_v)
            z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
            D_fake_gauss = Disc(z_fake_gauss)

            G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
            test_gen_loss += G_loss.item()


    test_len = len(test_loader.dataset)
    return test_loss, test_rec_loss / test_len, test_disc_loss / test_len, test_gen_loss / test_len, test_labels



def train_validate(encoder, decoder, Disc, train_loader, val_loader, args, optim_encoder, optim_decoder, optim_D, optim_encoder_reg,ae_criterion, Train):

    writer = SummaryWriter()
    train_loss = []
    val_loss = []

    train_rec_loss = 0
    train_disc_loss = 0
    train_gen_loss = 0

    val_rec_loss = 0
    val_disc_loss = 0
    val_gen_loss = 0

    device= args.device
    EPS = args.EPS
    train_labels=[]
    val_labels = []

    for epoch in range(args.num_epoch):
        for i, (data, labels) in enumerate(train_loader):
            encoder.train()
            decoder.train()
            Disc.train()

            """ Reconstruction loss """
            for p in Disc.parameters():
                p.requires_grad = False

            real_data_v = autograd.Variable(data).to(device)
            train_labels.append(labels)
            enc_mu, enc_var = encoder(real_data_v)
            encoding = reparameterize(enc_mu,enc_var)
            fake = decoder(encoding)
            ae_loss = ae_criterion(fake, real_data_v)

            train_rec_loss += ae_loss.item()
            train_loss.append(ae_loss.item())

            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            ae_loss.backward()
            optim_encoder.step()
            optim_decoder.step()

            """ Discriminator loss """
            encoder.eval()
            z_real_gauss = autograd.Variable(torch.randn(data.size()[0], 32,22,61) * 5.).to(device)
            D_real_gauss = Disc(z_real_gauss)

            mu_fake_gauss, var_fake_gauss  = encoder(real_data_v)
            z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
            D_fake_gauss = Disc(z_fake_gauss)

            D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
            train_disc_loss += D_loss.item()

            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()

            """ Generator loss """
            encoder.train()

            mu_fake_gauss, var_fake_gauss = encoder(real_data_v)
            z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
            D_fake_gauss = Disc(z_fake_gauss)

            G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
            train_gen_loss += G_loss.item()

            optim_encoder_reg.zero_grad()
            G_loss.backward()
            optim_encoder_reg.step()

            if i % 20 == 0:
                print ('Train [%d]/[%d], Recon. loss: %.4f, Discriminator loss :%.4f , Generator loss:%.4f'
                        %(epoch, i, ae_loss.item(), D_loss.item(), G_loss.item()))


        for i, (data, labels) in enumerate(val_loader):
            encoder.eval()
            decoder.eval()
            Disc.eval()

            with torch.no_grad():

                """ Reconstruction loss """
                for p in Disc.parameters():
                    p.requires_grad = False

                real_data_v = autograd.Variable(data).to(device)
                val_labels.append(labels)
                enc_mu, enc_var = encoder(real_data_v)
                encoding = reparameterize(enc_mu, enc_var)
                fake = decoder(encoding)
                ae_loss = ae_criterion(fake, real_data_v)

                val_rec_loss += ae_loss.item()
                val_loss.append(ae_loss.item())


                """ Discriminator loss """
                encoder.eval()
                z_real_gauss = autograd.Variable(torch.randn(data.size()[0], 32, 22, 61) * 5.).to(device)
                D_real_gauss = Disc(z_real_gauss)

                mu_fake_gauss, var_fake_gauss = encoder(real_data_v)
                z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
                D_fake_gauss = Disc(z_fake_gauss)

                D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
                val_disc_loss += D_loss.item()


                mu_fake_gauss, var_fake_gauss = encoder(real_data_v)
                z_fake_gauss = reparameterize(mu_fake_gauss, var_fake_gauss)
                D_fake_gauss = Disc(z_fake_gauss)

                G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
                val_gen_loss += G_loss.item()

                if i % 100 == 0:
                    print ('Val [%d]/[%d], Recon. loss: %.4f, Discriminator loss :%.4f , Generator loss:%.4f'
                            %(epoch, i, ae_loss.item(), D_loss.item(), G_loss.item()))

        writer.add_scalars("Train Rec Loss", {'train': train_rec_loss, 'val': val_rec_loss}, epoch)
        writer.add_scalars("Train Disc Loss", {'train': train_disc_loss, 'val': val_disc_loss}, epoch)
        writer.add_scalars("Train Gen Loss", {'train': train_gen_loss, 'val': val_gen_loss}, epoch)


    torch.save(encoder.state_dict(), args.model_path + 'encoder_epoch'+str(args.num_epoch)+'.pt')
    torch.save(decoder.state_dict(), args.model_path +'decoder_epoch'+str(args.num_epoch)+'.pt')
    torch.save(Disc.state_dict(), args.model_path +'disc_epoch'+str(args.num_epoch)+'.pt')

    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)

    return train_loss, train_rec_loss /train_len  , train_disc_loss /train_len , train_gen_loss /train_len, train_labels, val_loss, val_rec_loss /val_len  , val_disc_loss /val_len , val_gen_loss /val_len, val_labels


