train_losses = []
test_losses = []

model = mnist1_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if cuda else "cpu")

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
            if not isinstance(model, VAE3):
                sample = model.decode(sample).cpu()
            else:
                z2 = torch.randn(64, 50).to(device)
                sample = model.decode_for_testing(z2).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
print("Training finished")
logging.info("training finished")