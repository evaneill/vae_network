# Initialize a model and data loaders
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=test_batch_size, shuffle=True, **kwargs)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = mnist2_model().to(device)

# Call the training shenanigans
if torch.cuda.is_available(): 
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}',exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/samples',exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/recons',exist_ok=True)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
for r in range(num_rounds+1):
    current_round_lr = learning_rate*(10**(-r/7))
    optimizer = optim.Adam(model.parameters(), lr=current_round_lr)
    print(f"====== About to train for {3**r} epochs in round {r} with learning rate {round(current_round_lr,7)}========")
    logging.info(f"====== About to train for {3**r} epochs in round {r} with learning rate {round(current_round_lr,7)}========")
    for epoch in range(3**r):
        train(r,epoch,optimizer)
    with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}.pt','wb') as f:
        print(datetime.datetime.now())
        logging.info(datetime.datetime.now())
        torch.save(model,f)

    
    # _test(r,epoch)
    with torch.no_grad():
        sample = torch.randn(64, 50).to(device)
        sample = model.decode(sample,test=True).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    f'{model_type}_{data_name}_K{K}_M{batch_size}/samples/sample_' +str(r)+'_'+ str(epoch) + '.png')
                
_test(r,epoch)
with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_{r}_{epoch}.pt','wb') as f:
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    torch.save(model,f)
print("Training finished")
logging.info("Training finished")