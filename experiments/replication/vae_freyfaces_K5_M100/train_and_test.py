def train(epoch,collect_grad=False):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        # (B, 1, F1, F2) (e.g.
        data = data.to(device)
        optimizer.zero_grad()

        #recon_batch, mu, logvar = model(data)
        #loss = loss_function(recon_batch, data, mu, logvar)
        recon_batch, loss = model.compute_loss_for_batch(data, model)
        with detect_anomaly():
          loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if collect_grad==True and batch_idx==0:
            mu_grad = model.fc21.weight.grad.clone()
            output_grad = model.fc4.weight.grad.clone()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch,  train_loss / len(train_loader.dataset)))
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch,  train_loss / len(train_loader.dataset)))
    if collect_grad==True:
        # gradient in first minibatch of epoch of mu weights of encoder, and output layer weights of decoder
        return mu_grad, output_grad

# pycharm thinks that I want to run a test whenever I define a function that has 'test' as prefix
# this messes with running the model and is the reason why the function is called _test
def _test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, [data] in enumerate(test_loader):
            data = data.to(device)
            (recon_batch,_), mu, logvar = model(data)
            _, loss = model.compute_loss_for_batch(data, model, K=5000, test=True)
            test_loss += loss.item()
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1,1,28,20),
                                      recon_batch.view(test_batch_size, 1, 28, 20)[:n]])
                save_image(comparison.cpu(),
                         f'{model_type}_{data_name}_K{K}_M{batch_size}/recons/reconstruction_' +'_'+ str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    #test_loss *= 5000
    print('====> Epoch {}: Test set loss: {:.4f}'.format(epoch,test_loss))
    logging.info('====> Epoch {}: Test set loss: {:.4f}'.format(epoch,test_loss))