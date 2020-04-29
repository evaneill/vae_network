# Initialize a model and data loaders
train_loader = DataLoader(TensorDataset(data_train_t),batch_size=batch_size,shuffle=True,pin_memory=True)
test_loader = DataLoader(TensorDataset(data_test_t),batch_size=test_batch_size,shuffle=True,pin_memory=True)

device = torch.device('cuda')
model = freyface_model().to(device)

# Call the training shenanigans
if torch.cuda.is_available(): 
    print("Training on GPU")
    logging.info("Training on GPU")

os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}',exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/samples',exist_ok=True)
os.makedirs(f'{model_type}_{data_name}_K{K}_M{batch_size}/recons',exist_ok=True)

print(datetime.datetime.now())
logging.info(datetime.datetime.now())
if collect_grads==True:
    mu_grads=[]
    output_grads=[]

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"====== About to train for {epochs} epochs with learning rate {learning_rate}========")
logging.info(f"====== About to train for {epochs} epochs with learning rate {learning_rate}========")
for e in range(epochs):
    train(e,collect_grad=collect_grads)
    if e % log_test_value==0:  
        with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}_{e}.pt','wb') as f:
            print(datetime.datetime.now())
            logging.info(datetime.datetime.now())
            torch.save(model,f)
        _test(e)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            (sample, _) = model.decode(sample,test=True)
            sample = sample.cpu()
            save_image(sample.view(64, 1, 28, 20),
                        f'{model_type}_{data_name}_K{K}_M{batch_size}/samples/sample_' + str(e) + '.png')    
    
                
_test(e)
with open(f'{model_type}_{data_name}_K{K}_M{batch_size}/{model_type}_{data_name}_K{K}_M{batch_size}.pt','wb') as f:
    print(datetime.datetime.now())
    logging.info(datetime.datetime.now())
    torch.save(model,f)
print("Training finished")
logging.info("Training finished")