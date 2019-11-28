import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch

device = torch.device('cpu')

nn_img_size = 32
num_classes = 3
learning_rate = 0.0001
num_epochs = 500
batch_size = 4

loss_mode = 'crossentropy' 

loss_train_hist = []

##################################################
## Please implement a two layer neural network  ##
## A3.2
## @NB https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
##     https://www.youtube.com/watch?v=7qYtIveJ6hU
##     https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
##     https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
##################################################

def relu(x):
    """ReLU activation function"""
    return torch.clamp(x, min=0.0)

def relu_derivative(output):
    """derivative of the ReLU activation function"""
    output[output <= 0] = 0
    output[output>0] = 1
    return output

def softmax(z):
    """softmax function to transform values to probabilities"""
    z -= z.max()
    z = torch.exp(z)
    sum_z = z.sum(1, keepdim=True)
    return z / sum_z 

def loss_mse(activation, y_batch):
    """mean squared loss function"""
    # use MSE error as loss function 
    # Hint: the computed error needs to get normalized over the number of samples
    loss = (activation - y_batch).pow(2).sum() 
    mse = 1.0 / activation.shape[0] * loss
    return mse

def loss_crossentropy(activation, y_batch):
    """cross entropy loss function"""
    batch_size = y_batch.shape[0]
    loss = ( - y_batch * activation.log()).sum() / batch_size
    return loss

def loss_deriv_mse(activation, y_batch):
    """derivative of the mean squared loss function; derivative of the loss function w.r.t. to a2 """
    dCda2 = (1 / activation.shape[0]) * (activation - y_batch)
    return dCda2

def loss_deriv_crossentropy(activation, y_batch):
    """derivative of the mean cross entropy loss function"""
    batch_size = y_batch.shape[0]
    dCda2 = activation
    dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
    dCda2 /= batch_size
    return dCda2

def setup_train():
    """train function"""
    # load and resize train images in three categories
    # cars = 0, flowers = 1, faces = 2 ( true_ids )
    train_images_cars = glob.glob('./images/db/train/cars/*.jpg')
    train_images_flowers = glob.glob('./images/db/train/flowers/*.jpg')
    train_images_faces = glob.glob('./images/db/train/faces/*.jpg')
    train_images = [train_images_cars, train_images_flowers, train_images_faces]
    num_rows = len(train_images_cars)+len(train_images_flowers) +len(train_images_faces)
    X_train = torch.zeros((num_rows, nn_img_size*nn_img_size))
    y_train = torch.zeros((num_rows, num_classes))

    counter = 0
    for (label, fnames) in enumerate(train_images):
        for fname in fnames:
            print(label, fname)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (nn_img_size, nn_img_size) , interpolation=cv2.INTER_AREA)

            # print( label, " -- ", fname, img.shape)

            # fill matrices X_train - each row is an image vector
            # y_train - one-hot encoded, put only a 1 where the label is correct for the row in X_train
            y_train[counter, label] = 1
            X_train[counter] = torch.from_numpy(img.flatten().astype(np.float32))
            
            counter += 1

    # print(y_train)
    return X_train, y_train

def forward(X_batch, y_batch, W1, W2, b1, b2):
    """forward pass in the neural network """

    m1 = torch.mm(X_batch, W1) + b1
    a1 = relu(m1) # first activation
    m2 = torch.mm(a1, W2) + b2
    a2 = relu(m2) # second activation
    loss = loss_mse(a2, y_batch) # calculate the loss
    
    return loss, a2, a1

def backward(X_batch, y_batch, W1, W2, a1, a2, b1, b2):
    # Zx = mx
    #dCdW2 = dCdA2 * dA2dZ2 * dZ2dW2 weight before Output
    #dCdW1 = dCdA2 * dA2dZ2 * dZ2dA1 * dA1dZ1 * dZ1dW1 weight before Hidden
    Z1 = torch.mm(X_batch, W1) + b1
    Z2 = torch.mm(a1, W2) + b2

    dCdA2 = loss_deriv_mse(a2, y_batch) # loss derivative w.r.t. activation before; dCdZ2
    #dA2dZ2 = relu_derivative(Z2) # activation derivative w.r.t. hidden layer output
    dZ2dA1 = W2
    dA1dZ1 = relu_derivative(Z1)
    dZ1dW1 = X_batch
    dZ2dW2 = a1 # computed in forward pass
    #dCdA1 = torch.mm(W2.T, dCdA2)
    #dCdZ1 = dCdA1 * relu_derivative(Z1)
    tmp = torch.mm(dCdA2, dZ2dA1.T) * dA1dZ1
    tmp2 = torch.mm(dCdA2, dZ2dA1.T) * dA1dZ1

    dCdW1 = torch.mm(dZ1dW1.T, tmp) # weight before hidden layer
    dCdW2 = torch.mm(dZ2dW2.T, dCdA2) # weight before output layer
    dCdb1 = torch.sum(tmp2) # torch.mm(W2, dCdA2.T) * relu_derivative(Z1)
    dCdb2 = torch.sum(dCdA2) # bias before the output layer

    return dCdW1, dCdW2, (1/X_batch.shape[0]) * dCdb1, (1/a2.shape[0]) * dCdb2

def train(X_train, Y_train):
    """ train procedure """
    # for simplicity of this execise you don't need to find useful hyperparameter
    # I've done this for you already and every test image should work for the
    # given very small trainings database and the following parameters.
    h = 1500
    std = 0.001
    inSize = nn_img_size * nn_img_size
    print('inSize', inSize)
 
    # initialize W1, W2, b1, b2 randomly
    # Note: W1, W2 should be scaled by variable std
    W1 = std * torch.randn(inSize, h)
    W2 = std * torch.randn(h, num_classes)
    b1, b2 = torch.randn(1, h), torch.randn(1, num_classes)

    # run for num_epochs
    for i in range(num_epochs):

        X_batch = None
        Y_batch = None

        # use only a batch of batch_size of the training images in each run
        # sample the batch images randomly from the training set
        idx = torch.randint(low = 0, high = 20, size = (batch_size,))
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]

        # forward pass for two-layer neural network using ReLU as activation function
        loss, a2, a1 = forward(X_batch, Y_batch, W1, W2, b1, b2)
        
        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)
        
        #if i % 10 == 0:
        #    print("iteration %d: loss %f" % (i, loss))

        # backward pass
        dCdW1, dCdW2, dCdb1, dCdb2 = backward(X_batch, Y_batch, W1, W2, a1, a2, b1, b2)

        # depending on the derivatives of W1, and W2 regaring the cost/loss
        # we need to adapt the values in the negative direction of the 
        # gradient decreasing towards the minimum
        # we weight the gradient by a learning rate

        W2 += dCdW2 * learning_rate
        W1 += dCdW1 * learning_rate
        #print(W1.shape)
        #print(W2.shape)

        b2 += dCdb2 * learning_rate
        b1 += dCdb1 * learning_rate

        #print(b1.shape)
        #print(b2.shape)
        
    return W1, W2, b1, b2

X_train, y_train = setup_train()
print('before train: ', y_train)
W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and 
# run prediction by computing the forward pass
test_images = []
test_images.append( (cv2.imread('./images/db/test/flower.jpg', cv2.IMREAD_GRAYSCALE), 1) )
test_images.append( (cv2.imread('./images/db/test/car.jpg', cv2.IMREAD_GRAYSCALE), 0) )
test_images.append( (cv2.imread('./images/db/test/face.jpg', cv2.IMREAD_GRAYSCALE), 2) )
y_test = torch.zeros((3, num_classes))
y_test[0] = torch.Tensor([0, 1, 0])
y_test[1] = torch.Tensor([1, 0, 0])
y_test[2] = torch.Tensor([0, 0, 1])
print(y_test)
for ti in test_images:
#    print(ti[0], "\n", ti[1])
    resized_ti = cv2.resize(ti[0], (nn_img_size, nn_img_size) , interpolation=cv2.INTER_AREA)
    X_test = torch.from_numpy(resized_ti.reshape(1,-1))
    loss_test, a1_test, a2_test = forward(X_test, y_test, W1, W2, b1, b2)
    print("Test output (values / pred_id / true_id):", a2_test, np.argmax(a2_test), ti[1])
    

# print("------------------------------------")
# print("Test model output Weights:", W1, W2)
# print("Test model output bias:", b1, b2)


plt.title("Training Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1,num_epochs +1),loss_train_hist,label="Train")
plt.ylim((0,3.))
plt.xticks(np.arange(1, num_epochs+1, 50.0))
plt.legend()
plt.show()
plt.savefig("simple_nn_train.png")

