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
##     https://github.com/geeksnome/machine-learning-made-easy/blob/master/backpropogation.py
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

def backwardSimple(X_batch, y_batch, a1, a2, W1, W2):
    """ a2 == output """
    output_error = loss_mse(a2, y_batch) # aka loss
    output_delta = output_error * relu_derivative(a2)

    # hidden layer
    dCda1 = torch.mm(output_delta, W2.t()) # contribution of hidden layer weights to output error
    a1_delta = dCda1 * relu_derivative(a1)

    W1 += torch.mm(X_batch.t(), a1_delta) * learning_rate
    W2 += torch.mm(a2.t(), output_delta) * learning_rate

    return W1, W2

def backward(a2, a1, X_batch, y_batch, W1, W2, b1, b2):
    """backward pass in the neural network """
    # Implement the backward pass by computing
    # the derivative of the complete function
    # using the chain rule as discussed in the lecture

    dCda2 = loss_deriv_mse(a2, y_batch) # derivative of loss function
    m2 = torch.mm(a1, W2) + b2
    dCdm2 = relu_derivative(m2)

    dm2da1 = W2 # impact of a1 on m2
    da1dm1 = relu_derivative(torch.mm(X_batch, W1) + b1)
    dCda1 = torch.mm(W2.t(), dCdm2)
    dCdm1 = torch.mm(dCda1, da1dm1)

    dCdW1 = torch.mm(X_batch, dCdm1)

#    dCdW2 = torch.mm(dCda2, W2.t()) # derivative of loss w.r.t W2

    #dCda1 =
    #dCdW1 =
     

    
    # function should return 4 derivatives with respect to
    # W1, W2, b1, b2
    return dCdW1, dCdW2, dCdb1, dCdb2

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
        print(idx)
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]

        # forward pass for two-layer neural network using ReLU as activation function
        loss, a2, a1 = forward(X_batch, Y_batch, W1, W2, b1, b2)
        
        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)
        
        #if i % 10 == 0:
        #    print("iteration %d: loss %f" % (i, loss))

        # backward pass
        #error_out = loss * relu_derivative(a2)
        #dCdW1, dCdW2, dCdb1, dCdb2 = backward(a2, a1, X_batch, Y_batch, W1, W2, b1, b2) 
        W1, W2 = backwardSimple(X_batch, Y_batch, a1, a2, W1, W2)    
        # print("dCdb2.shape:", dCdb2.shape, dCdb1.shape)

        # depending on the derivatives of W1, and W2 regaring the cost/loss
        # we need to adapt the values in the negative direction of the 
        # gradient decreasing towards the minimum
        # we weight the gradient by a learning rate
        #W2 += torch.mm(a1.t(), ) * learning_rate
        #W1 += torch.mm(X_batch.t(), ) * learning_rate

        #b2 += .sum() * learning_rate
        #b1 += .sum() * learning_rate
        
    return W1, W2, b1, b2

X_train, y_train = setup_train()
W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and 
# run prediction by computing the forward pass
test_images = []
test_images.append( (cv2.imread('./images/db/test/flower.jpg', cv2.IMREAD_GRAYSCALE), 1) )
test_images.append( (cv2.imread('./images/db/test/car.jpg', cv2.IMREAD_GRAYSCALE), 0) )
test_images.append( (cv2.imread('./images/db/test/face.jpg', cv2.IMREAD_GRAYSCALE), 2) )

for ti in test_images:
    resized_ti = cv2.resize(ti[0], (nn_img_size, nn_img_size) , interpolation=cv2.INTER_AREA)
    x_test = resized_ti.reshape(1,-1)
    # YOUR CODE HERE 
    # convert test images to pytorch
    # do forward pass depending mse or softmax
    # print("Test output (values / pred_id / true_id):", a2_test, np.argmax(a2_test), ti[1])

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

