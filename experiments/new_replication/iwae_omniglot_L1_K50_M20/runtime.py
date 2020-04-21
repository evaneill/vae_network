# Initialize a model and data loaders
train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True,pin_memory=True)

device = torch.device('cuda')
model = omniglot1_model().to(device)

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
        train(r,epoch, optimizer)
    with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_LR0001.pt','wb') as f:
        print(datetime.datetime.now())
        logging.info(datetime.datetime.now())
        torch.save(model,f)

    
    _test(r,epoch)
    with torch.no_grad():
        sample = torch.randn(64, 200).to(device)
        sample = model.decode(sample,test=True).cpu()
        save_image(sample.view(64, 1, 28, 28),
                    f'{model_type}_{data_name}_K{K}_M{batch_size}/samples/sample_' +str(r)+'_'+ str(epoch) + '.png')
                
_test(r,epoch)
with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_LR0001_{r}_{epoch}.pt','wb') as f:
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    torch.save(model,f)
print("Training finished")
logging.info("Training finished")
