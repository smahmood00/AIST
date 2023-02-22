from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# define model parameters
BATCH_SIZE = 64
IMAGE_DIM = 227  # pixels
NUM_CLASSES = 1000
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use

#data directories
INPUT_ROOT_DIR = 'AIST'
TRAIN_IMG_DIR = 'data/train'
VAL_IMG_DIR = 'data/val'
TEST_IMG_DIR = 'data/test'
OUTPUT_DIR = 'alexnet_data_out'

def create_dataset():
    transformation = transforms.Compose([
            transforms.CenterCrop(IMAGE_DIM),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root = TRAIN_IMG_DIR, transform = transformation)
    val_dataset = ImageFolder(root = VAL_IMG_DIR,  transform = transformation)
    #test_dataset = ImageFolder(root = TEST_IMG_DIR, transform = transformation)
    
    return train_dataset, val_dataset
  
def create_dataloader(train_dataset,val_dataset):
    trainloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)

    valloader = data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
    print('Dataloader created')

    return trainloader, valloader