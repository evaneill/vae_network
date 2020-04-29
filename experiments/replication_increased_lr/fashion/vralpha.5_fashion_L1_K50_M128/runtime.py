train_losses = []
test_losses = []

device = torch.device("cuda" if cuda else "cpu")

model = mnist1_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=test_batch_size, shuffle=True, **kwargs)

if torch.cuda.is_available(): 
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs('results', exist_ok=True)
print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for epoch in range(1, epochs + 1):
    train(epoch)
    if epoch % testing_frequency == 1:
        _test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 50).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
print("Training finished")
logging.info("training finished")