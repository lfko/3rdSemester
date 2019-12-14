import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.autograd import Variable


def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction
    """

    images = []

    # TODO read each image in path as numpy.ndarray and append to images
    # Useful functions: lib.list_directory(), matplotlib.image.imread(), numpy.asarray()

    files = os.listdir(path)
    files.sort()
    for cur in files:
        if not cur.endswith(file_ending):
            continue

        try:
            image = mpl.image.imread(path + cur)
            img_mtx = np.asarray(image, dtype="float64")
            images.append(img_mtx)
        except:
            continue

    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        '''
        self.encoder = nn.Sequential(
            #nn.Linear(1 * 96 * 118, 512), nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1, stride = 1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 3, 1, 1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(inplace = True),
            nn.Upsample(2, 2),
            nn.Conv2d(8, 8, 3, 1, 1), nn.ReLU(inplace = True),
            nn.Upsample(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(inplace = True),
            nn.Upsample(2, 2),
            nn.Conv2d(16, 1, 3, 1, 1) # should restore the input image
        )
        '''
        self.encoder = nn.Sequential(
            nn.Linear(98 * 116, 512), nn.ReLU(inplace = True),
            nn.Linear(512, 64), nn.ReLU(inplace = True),
            nn.Linear(64, 16), nn.ReLU(inplace = True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(inplace = True),
            nn.Linear(128, 256), nn.ReLU(inplace = True),
            nn.Linear(256, 98 * 116),
        )

    def forward(self, x):
        in_size = x.size(0)
        print(in_size)
        #x = x.view(1, -1)

        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__ == '__main__':

    # pictures should be 98x116
    images, x, y = load_images('../data/train/')

    # setup data matrix
    D = np.zeros((len(images), images[0].size), dtype=np.float32)
    mean_data = np.mean(D)
    for i in range(len(images)):
        D[i, :] = images[i].flatten() # every image is a 11368 long vector now

    for i in range(D.shape[0]):
        D[i, :] -= mean_data

    #print('before: ', D[0])
    #print('mean:', np.mean(D))
    #D = D - np.mean(D)
    #print('after: ', D[0])
    #print(D.shape)

    num_epochs = 2000
    batch_size = 50
    learning_rate = 0.01


    data = torch.from_numpy(D)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-05)

    for epoch in range(num_epochs):
        data = Variable(data)
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        MSE_loss = nn.MSELoss()(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data, MSE_loss.data))

    # now we use the nn model to reconstruct test images
    # and measure their reconstruction error

    images_test, x, y = load_images('../data/test/')
    D_test = np.zeros((len(images_test), images_test[0].size), dtype=np.float32)
    for i in range(len(images_test)):
        D_test[i, :] = images_test[i].flatten()

    for i in range(D_test.shape[0]):
        D_test[i, :] -= mean_data

    data_test = torch.from_numpy(D_test)

    errors = []
    for i, test_image in enumerate(images_test):

        # evaluate the model using data_test samples i
        img_reconst = model(data_test[i])
        img_reconst = img_reconst.data.numpy()
        img_reconst += mean_data
        # add the mean to the predicted/reconstructed image
        # and reshape to size (116,98)
        img_reconst = img_reconst.reshape((116, 98))

        # uncomment
        error = np.linalg.norm(images_test[i] - img_reconst)
        errors.append(error)
        print("reconstruction error: ", error)

    grid = plt.GridSpec(2, 9)

    plt.subplot(grid[0, 0:3])
    plt.imshow(images_test[8], cmap='Greys_r')
    plt.xlabel('Original person')

    pred = model(data_test[8, :])
    pred_np = pred.data.numpy()
    pred_np += mean_data
    img_reconst = pred_np.reshape((116, 98))
    plt.subplot(grid[0, 3:6])
    plt.imshow(img_reconst, cmap='Greys_r')
    plt.xlabel('Reconstructed image')

    plt.subplot(grid[0, 6:])
    plt.plot(np.arange(len(images_test)), errors)
    plt.xlabel('Errors all images')

    plt.savefig("pca_ae_solution.png")
    plt.show()

    print("Mean error", np.asarray(errors).mean())
