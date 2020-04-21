# Initialize a model and data loaders
train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True,pin_memory=True)

device = torch.device('cuda')
model = omniglot2_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Call the training shenanigans
if torch.cuda.is_available(): 
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs('results/',exist_ok=True)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for epoch in range(1, epochs + 1):
    train(epoch)
    if epoch % testing_frequency == 1:
        _test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 50).to(device)
            sample = model.decode(sample,test=True).cpu()
            save_image(sample.view(64, 1, 28, 28),
                        'results/sample_' + str(epoch) + '.png')
print(datetime.datetime.now())
print("Training finished")
logging.info("Training finished")