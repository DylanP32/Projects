import cupy as cp
from matplotlib import pyplot as plt
import time


# Perform gradient descent to train a model 
def gradient_descent(input_data, true_labels, learning_rate, epochs):

    hidden_weights, hidden_bias, output_weights, output_bias = init_params()

    m_num_examples = input_data.shape[1]

    for i in range(epochs):

        # predict
        Z_hidden, A_hidden, Z_output, A_output = forward_prop(hidden_weights, hidden_bias, output_weights, output_bias, input_data)

        # learn
        dW_hidden, db_hidden, dW_output, db_output = backward_prop(Z_hidden, A_hidden, Z_output, A_output, hidden_weights, output_weights, input_data, true_labels, m_num_examples)

        # adapt
        hidden_weights, hidden_bias, output_weights, output_bias = update_params(hidden_weights, hidden_bias, output_weights, output_bias, dW_hidden, db_hidden, dW_output, db_output, learning_rate)

        # report progress
        if i % 10 == 0:
            print("Epoch: ", i)
            predictions = get_predictions(A_output)
            print(get_accuracy(predictions, true_labels))

    return hidden_weights, hidden_bias, output_weights, output_bias


# Initialize randomized parameters for the neural network
def init_params():

    # "He initialization" for weights, using sqrt(2/n) where n is the number of input features
    # biases are initialized with "He initialization" as well for consistency with the weights

    hidden_weights = cp.random.normal(size=(256, 784)) * cp.sqrt(2./784)      # weights from 784 neurons to 256 neurons
    hidden_bias = cp.random.normal(size=(256, 1)) * cp.sqrt(2./256)             # bias for 256 neurons

    output_weights = cp.random.normal(size=(10, 256)) * cp.sqrt(2. / 256)       # weights from 256 neurons to 10 output neurons
    output_bias = cp.random.normal(size=(10, 1)) * cp.sqrt(2. / 10)             # bias for 10 output neurons

    return hidden_weights, hidden_bias, output_weights, output_bias


# Forward and backward propagation functions
def forward_prop(hidden_weights, hidden_bias, output_weights, output_bias, input_data):

    Z_hidden = hidden_weights.dot(input_data) + hidden_bias      # hidden layer pre activation
    A_hidden = ReLU(Z_hidden)                                    # hidden layer post activation

    Z_output = output_weights.dot(A_hidden) + output_bias          # output layer pre activation
    A_output = softmax(Z_output)                                 # output layer post activation

    return Z_hidden, A_hidden, Z_output, A_output
    
def backward_prop(Z_hidden, A_hidden, Z_output, A_output, hidden_weights, output_weights, input_data, true_labels, m_num_examples):

    # kept Z_output and hidden_weights because they could be used in other algorithms.
    # but they are not used in this version of a backward pass

    label_one_hot = one_hot(true_labels)

    dZ_output = A_output - label_one_hot                                        # output layer gradient loss
    dW_output = 1 / m_num_examples * dZ_output.dot(A_hidden.T)                  # output layer weight gradient
    db_output = 1 / m_num_examples * cp.sum(dZ_output, axis=1, keepdims=True)   # output layer bias gradient

    dZ_hidden = output_weights.T.dot(dZ_output) * ReLU_deriv(Z_hidden)           # hidden layer gradient loss
    dW_hidden = 1 / m_num_examples * dZ_hidden.dot(input_data.T)                # hidden layer weight gradient
    db_hidden = 1 / m_num_examples * cp.sum(dZ_hidden, axis=1, keepdims=True)   # hidden layer bias gradient

    return dW_hidden, db_hidden, dW_output, db_output


# Utility functions for propogation
def one_hot(Y):
    one_hot_Y = cp.zeros((Y.size, Y.max().item() + 1))
    one_hot_Y[cp.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def ReLU_deriv(Z):
    return Z > 0

def ReLU(Z): # hidden layer activation
    return cp.maximum(Z, 0)

def softmax(Z): # output layer activation
    A = cp.exp(Z) / cp.sum(cp.exp(Z), axis=0)
    return A


# Updates parameters by performing a single step of gradient descent
def update_params(hidden_weights, hidden_bias, output_weights, output_bias, dW_hidden, db_hidden, dW_output, db_output, learning_rate):

    updated_h_weights = hidden_weights - learning_rate * dW_hidden
    updated_h_bias = hidden_bias - learning_rate * db_hidden  

    updated_o_weights = output_weights - learning_rate * dW_output
    updated_o_bias = output_bias - learning_rate * db_output

    return updated_h_weights, updated_h_bias, updated_o_weights, updated_o_bias


# Utility functions for predictions and accuracy calculation
def get_predictions(A_output):
    return cp.argmax(A_output, 0)

def get_accuracy(predictions, true_labels):
    print(predictions, true_labels)
    return cp.sum(predictions == true_labels) / true_labels.size

# Testing functions

# batch testing
def make_predictions(input_data, hidden_weights, hidden_bias, output_weights, output_bias): 
    _, _, _, A_output = forward_prop(hidden_weights, hidden_bias, output_weights, output_bias, input_data)
    predictions = get_predictions(A_output)
    return predictions

# single input testing
def test_prediction(index, hidden_weights, hidden_bias, output_weights, output_bias, train_features, train_labels): 
    current_image = train_features[:, index, None]
    prediction = make_predictions(train_features[:, index, None], hidden_weights, hidden_bias, output_weights, output_bias)
    label = train_labels[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    current_image = cp.asnumpy(current_image)  # for cupy
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#################################################### MAIN ####################################################

# reminders:
# 1. Features are the literal pixel values of the images
# 2. dev set is for initial testing and validation
# 3. training set is for training the model
# 4. The model is trained on the training set and evaluated on the dev set
# 5. Veriable descriptions:
#   - Z is the pre-activation value
#   - A is the output of the activation function (so post-activation)
#   - W is the weight matrix
#   - b is the bias vector
#   - dZ is the gradient of the loss with respect to Z
#   - dW is the gradient of the loss with respect to W (weights)
#   - db is the gradient of the loss with respect to b (biases)



# load the MNIST dataset
mnist_data = cp.loadtxt('mnist_train.csv', delimiter=',', skiprows=1, dtype=cp.float32)
m_num_examples, num_features = mnist_data.shape
cp.random.shuffle(mnist_data) # shuffle before splitting into dev and training sets


# dev set (first 1000 examples)
dev_data_transposed = mnist_data[0:1000].T
dev_labels = dev_data_transposed[0].astype(cp.int32)            # dev labels 
dev_features = dev_data_transposed[1:num_features]              # dev features 
dev_features = dev_features / 255.                              # normalize dev pixel values


# training set (the remaining examples)
train_data_transposed = mnist_data[1000:m_num_examples].T
train_labels = train_data_transposed[0].astype(cp.int32)        # training labels
train_features = train_data_transposed[1:num_features]          # training features
train_features = train_features / 255.                          # normalize training pixel values
_, num_train_samples = train_features.shape


# set hyperparameters
learning_rate = 0.01 # a.k.a. alpha
epochs = 1000


# perform gradient descent to train the model
time_start = time.time()
print("\nTraining the model...\n")
hidden_weights, hidden_bias, output_weights, output_bias = gradient_descent(train_features, train_labels, learning_rate, epochs)
print("\nModel trained.\n")
time_end = time.time()
print("Time taken to train the model: ", time_end - time_start, "seconds\n")


# test the model on training examples
print("\nTesting the model on training examples...\n")
for _ in range(9):
    test_prediction(cp.random.randint(0, 1000), hidden_weights, hidden_bias, output_weights, output_bias, train_features, train_labels)


# test the model on dev set
print("\nTesting the model on dev set...\n")
dev_predictions = make_predictions(dev_features, hidden_weights, hidden_bias, output_weights, output_bias)
print(get_accuracy(dev_predictions, dev_labels))