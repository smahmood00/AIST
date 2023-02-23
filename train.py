#import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

from datasets import create_dataset, create_dataloader #function from datasets.py 
from model import AlexNet #class from model.py 
from utils import save_model,save_plots, SaveBestModel #functions and class from utils.py 


# define pytorch device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using: ', device)

# define model parameters
NUM_EPOCHS = 100  # original paper 90
BATCH_SIZE = 64
#MOMENTUM = 0.9
#LR_DECAY = 0.0005
LR_INIT = 0.0001
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use

# modify this to point to your data directory
INPUT_ROOT_DIR = 'AIST'
TRAIN_IMG_DIR = 'data/train'
VAL_IMG_DIR = 'data/val'
TEST_IMG_DIR = 'data/test'
OUTPUT_DIR = 'outputs'
CHECKPOINT_DIR = OUTPUT_DIR   # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_loop(model,loss_fn,optimizer,dataloader):
    running_loss = 0
    size = len(dataloader.dataset)
    running_correct = 0
    for batch, (imgs, classes) in enumerate(dataloader):
        correct = 0
        imgs, classes = imgs.to(device), classes.to(device)

        # calculate the loss
        output = model(imgs)
        loss = loss_fn(output, classes)

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            correct =  (output.argmax(1) == classes).type(torch.float).sum().item() 
            running_correct += correct
         
        # log the information and add to tensorboard
        running_loss += loss.item()
        if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(imgs)
                #accuracy for the current batch
                acc = (correct / len(imgs)) 
        
                print(f"\tLoss: {loss:.4f} \tAccuracy {acc:.3f} \t[{current}/{size}]")
        
        
    
    epoch_train_accuracy = running_correct/len(dataloader)
    epoch_train_loss = running_loss/len(dataloader)
    
    return epoch_train_accuracy, epoch_train_loss 

def val_loop(model,loss_fn,dataloader):
    running_loss =0
    running_correct= 0
    model.eval()
    with torch.no_grad(): 
        for batch, (imgs, classes) in enumerate(dataloader):
            correct = 0
            imgs, classes = imgs.to(device), classes.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, classes)
            
            running_loss += loss.item()
            correct = (pred.argmax(1) == classes).type(torch.float).sum().item()
            running_correct += correct

            if batch % 10 == 0:
                #accuracy for the current batch
                acc = (correct / len(imgs)) 
                print(f"\tVal Loss: {loss.item():.4f} Val Accuracy: {acc:.4f}  ")

    epoch_val_accuracy = running_correct / len(dataloader)   
    epoch_val_loss = running_loss/len(dataloader)
 
    model.train()
    return epoch_val_accuracy, epoch_val_loss


if __name__ == '__main__':
    
    train_dataset , val_dataset = create_dataset()
    trainloader , valloader = create_dataloader(train_dataset,val_dataset)

    # create model
    model = AlexNet(num_classes=NUM_CLASSES).to(device)
    print(model)
    print('AlexNet created')

    # create optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR_INIT)
    print('Optimizer created')

    #instantiate the class to store the best model
    save_best_model = SaveBestModel()

    # multiply LR by 1 / 10 after every 30 epochs
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # print('LR Scheduler created')

    #loss function
    criterion = nn.CrossEntropyLoss()
    print('Loss function initialised')
    
    # start training
    print('Starting training...')

    train_loss_list = [], val_loss_list = []
    train_acc_list = [], val_acc_list = []
    for epoch in range(NUM_EPOCHS):
        #lr_scheduler.step()
        print(f"Epoch: {epoch+1}")
        train_accuracy,train_loss = train_loop(model,criterion,optimizer,trainloader)
        val_accuracy,val_loss = val_loop(model,criterion,valloader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        print(f"Epoch: {epoch+1} Avg Train Loss:{train_loss :.4f}  Train Accuracy: {train_accuracy:.3f}, Avg Val Loss:{val_loss:.4f}  Val Accuracy: {val_accuracy:.3f} \n")
        
        save_best_model(val_loss, epoch, model, optimizer, criterion)
        
        print('-'*50)

        save_model(epoch+1, model, optimizer, criterion,val_loss)

    save_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    
