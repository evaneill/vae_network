# Initialize a model and data loaders
model = omniglot1_model().to(device)

train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True)

# Call the training shenanigans
if torch.cuda.is_available(): 
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs('results', exist_ok=True)
model = omniglot1_model().to(device)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for i in range(num_rounds):
    current_round_lr = learning_rate*math.pow(10,-i/7)
    optimizer = optim.Adam(model.parameters(), lr=current_round_lr)
    print(f"========Current round LR: {current_round_lr}=======")
    logging.info(f"========Current round LR: {current_round_lr}=======")
    print(f"======== About to train for {3**i} epochs =========")
    logging.info(f"======== About to train for {3**i} epochs =========")
    for epoch in range(1,3**i + 1):
        train(i,epoch,optimizer)
        if epoch == 1 or epoch % 3**(i-1) ==0:
            _test(i,epoch)
            with torch.no_grad():
                z2 = torch.randn(64, 50).to(device)
                sample = model.decode(z2).cpu()
                save_image(sample.view(64, 1, 28, 28),
                            'results/sample_' +str(i)+'_'+ str(epoch) + '.png')

_test(i,epoch)
print(datetime.datetime.now())
logging.info(datetime.datetime.now())
print("Training finished")
logging.info("Training finished")