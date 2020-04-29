# train and test functions
def train(round_num,epoch,optimizer):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        recon_batch, _, _, loss = model.compute_loss_for_batch(data, model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logging.info('Round {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                round_num,epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Round {}, Epoch: {} Average loss: {:.4f}'.format(
          round_num,epoch,  train_loss / len(train_loader.dataset)))
    logging.info('====> Round {}, Epoch: {} Average loss: {:.4f}'.format(
          round_num,epoch,  train_loss / len(train_loader.dataset)))

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(round_num,epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            _, _, _, loss = model.compute_loss_for_batch(data, model, K=5000,test=True)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,28),
                                      recon_batch.view(test_batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(round_num)+'_'+str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Round {} Test set loss: {:.4f}'.format(round_num,test_loss))
    logging.info('====> Round {} Test set loss: {:.4f}'.format(round_num,test_loss))