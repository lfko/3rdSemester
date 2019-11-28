# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code
# https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/6
# https://towardsdatascience.com/build-a-fashion-mnist-cnn-pytorch-style-efb297e22582
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

use_cuda = torch.cuda.is_available()


root = './data'

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# images are 28 x 28 x 1
train_set = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

# hyperparameter
# TODO Find good hyperparameters
batch_size = 64
num_epochs = 15
learning_rate = .001
momentum = .05

# Load train and test data
data_loaders = {}
data_loaders['train'] = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
data_loaders['test'] = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

# implement your own NNs

class KindOfLeNet(nn.Module):
    def __init__(self):
        super(KindOfLeNet, self).__init__()
        # IN: 28 x 28 x 1
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride = 1, padding = 0), 
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        # OUT: 13 x 13 x 6
        self.l2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride = 1, padding = 0), 
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        # OUT: 7 x 7 x 16
        #self.l3 = nn.Sequential(
        #    nn.Conv2d(6, 16, 3, stride = 1, padding = 0), 
        #    nn.Tanh()
        #)
        # OUT: 10 x 10 x 16
        #self.l4 = nn.Sequential(
        #    nn.AvgPool2d(2, 2)
        #)
        # OUT: 5 x 5 x 16
        #self.l5 = nn.Sequential(
        #    nn.Conv2d(16, 120, 3), nn.ReLU()
        #)
        # OUT: 6 x 6 x 120
        
        self.classifier = nn.Sequential(
            nn.Linear(5 * 5 * 16, 64), nn.Tanh(),
            nn.Linear(64, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 10), nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        #x = self.l3(x)
        #x = self.l4(x)
        #x = self.l5(x)
        #x = x.view(5 * 5 * 16, -1) # TODO
        #print(x.size())
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def name(self):
        return "KindOfLeNet"

# http://personal.ie.cuhk.edu.hk/~ccloy/files/aaai_2015_target_coding_supp.pdf
class MyCNN(nn.Module):
    def __init__(self, classes = 10):
        super(MyCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2) # OUT: 32 x 14 x 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2) # OUT: 64 x 7 x 7
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2, 1) # OUT: 128 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 625), nn.ReLU(), nn.Linear(625, 10)
        )

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # flattening before FC
        return self.fc(x)

    def name(self):
        return 'You are using "MyCNN"'

# https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
class VGG16(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(2,2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(2, 2, 1)
        )
        # OUT: 256 x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 256 * 4 * 4)

        return self.fc(x)

    def name(self):
        return 'SomeOtherNet'


## training
model = KindOfLeNet() # Best val Acc: 0.7057
#model = VGG16() # Best val Acc: 0.8784
#model = MyCNN() # Best val Acc: 0.7798

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

criterion = nn.CrossEntropyLoss()

train_acc_history = []
test_acc_history = []

train_loss_history = []
test_loss_history = []


best_acc = 0.0
since = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'test']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(data_loaders[phase]):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print('{} Batch: {} of {}'.format(phase, batch_idx, len(data_loaders[phase])))

        epoch_loss = running_loss / len(data_loaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'test':
            test_acc_history.append(epoch_acc)
            test_loss_history.append(epoch_loss)
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)

    print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

acc_train_hist = []
acc_test_hist = []

acc_train_hist = [h.cpu().numpy() for h in train_acc_history]
acc_test_hist = [h.cpu().numpy() for h in test_acc_history]

plt.title("Validation/Test Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Accuracy")
plt.plot(range(1,num_epochs+1),acc_train_hist,label="Train")
plt.plot(range(1,num_epochs+1),acc_test_hist,label="Test")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

plt.title("Validation/Test Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation/Test Loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="Train")
plt.plot(range(1,num_epochs+1),test_loss_history,label="Test")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


examples = enumerate(data_loaders['test'])
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
  output = model(example_data)

categories = {
    0:	'T-shirt/top',
    1:	'Trouser',
    2:	'Pullover',
    3:	'Dress',
    4:	'Coat',
    5:	'Sandal',
    6:	'Shirt',
    7:	'Sneaker',
    8:	'Bag',
    9:	'Ankle boot'
}

for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Pred: {}".format(
      categories[output.data.max(1, keepdim=True)[1][i].item()]))
  plt.xticks([])
  plt.yticks([])
plt.show()

