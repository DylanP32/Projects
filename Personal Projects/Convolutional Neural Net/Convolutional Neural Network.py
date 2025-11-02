import cupy as cp


def init_params(kernel_size, conv1_size, conv2_size, input_size, hidden_size, OUTPUT_SIZE):

    # "He initialization" for weights, using sqrt(2/n) where n is the number of input features
    
    # conv layer parameters
    input_channels_L1 = 1 # 3 for RGB, 1 for grayscale
    conv1_input = input_channels_L1 * kernel_size
    input_channels_L2 = conv1_size
    conv2_input = input_channels_L2 * kernel_size
    
    # weights and biases for first layer of 3x3 kernels
    conv1_weights = cp.random.normal(size=(conv1_size, input_channels_L1, kernel_size), dtype=cp.float32) * cp.sqrt(2. / conv1_input)
    conv1_bias = cp.zeros(shape=(1, conv1_size, 1), dtype=cp.float32)
    
    # weights and biases for second layer of 3x3 kernels
    conv2_weights = cp.random.normal(size=(conv2_size, input_channels_L2, kernel_size), dtype=cp.float32) * cp.sqrt(2. / conv2_input)
    conv2_bias = cp.zeros(shape=(1, conv2_size, 1), dtype=cp.float32)
    

    # weights and biases from 3136 neurons to 256 hidden neurons. input layer/flattened layer
    hidden_weights = cp.random.normal(size=(hidden_size, input_size), dtype=cp.float32) * cp.sqrt(2. / input_size)
    hidden_bias = cp.zeros(hidden_size, dtype=cp.float32)
    
    # weights and biases from 256 hidden neurons to 10 output neurons
    output_weights = cp.random.normal(size=(OUTPUT_SIZE, hidden_size), dtype=cp.float32) * cp.sqrt(2. / hidden_size)
    output_bias = cp.zeros(OUTPUT_SIZE, dtype=cp.float32)

    return conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias

def update_params(conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias,
                  deriv_weight_hidden, deriv_bias_hidden, deriv_weight_output, deriv_bias_output, kernel_gradients2, deriv_conv2_bias,
                  kernel_gradients1, deriv_conv1_bias,
                  learning_rate, batch_size):

    # average gradients across the batch dimension
    
    # conv gradients
    avg_kernel_gradients1 = kernel_gradients1.sum(axis=0) / batch_size
    avg_kernel_gradients2 = kernel_gradients2.sum(axis=0) / batch_size
    
    # bias gradients
    avg_deriv_conv1_bias = deriv_conv1_bias.sum(axis=0) / batch_size
    avg_deriv_conv2_bias = deriv_conv2_bias.sum(axis=0) / batch_size
    avg_deriv_bias_hidden = deriv_bias_hidden.sum(axis=0) / batch_size
    avg_deriv_bias_output = deriv_bias_output.sum(axis=0) / batch_size

    # fully connected layer weight gradients
    avg_deriv_weight_output = (deriv_weight_output / batch_size)
    avg_deriv_weight_hidden = deriv_weight_hidden / batch_size

    # update parameters with gradient descent

    conv1_weights -= learning_rate * avg_kernel_gradients1
    conv1_bias -= learning_rate * avg_deriv_conv1_bias
    conv2_weights -= learning_rate * avg_kernel_gradients2
    conv2_bias -= learning_rate * avg_deriv_conv2_bias

    hidden_weights -= learning_rate * avg_deriv_weight_hidden
    hidden_bias -= learning_rate * avg_deriv_bias_hidden
    output_weights -= learning_rate * avg_deriv_weight_output
    output_bias -= learning_rate * avg_deriv_bias_output
    
    return conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias


def zero_padding(batch):
    
    # get feature map size and channels
    image_width = int(batch.shape[2] ** .5)
    input_channels = batch.shape[1]
    
    # prepare output array for padded feature maps
    padded_num_pixels = (image_width + 2) ** 2
    padded_image = cp.zeros((batch.shape[0], input_channels, padded_num_pixels), dtype=cp.float32)

    # copy map data into padded array
    for index, feature_map in enumerate(batch):
        for channel in range(input_channels):
            padded_image_i = image_width + 3
            feature_map_j = 0
            while feature_map_j < batch.shape[2]:
                padded_image[index, channel, padded_image_i:padded_image_i + image_width] = feature_map[channel, feature_map_j:feature_map_j + image_width]
                padded_image_i += image_width + 2
                feature_map_j += image_width

    return padded_image

def zero_unpadding(padded_batch):
    
    # get sizes of feature maps before and after padding
    padded_image_width = int(padded_batch.shape[2] ** .5)
    input_channels = padded_batch.shape[1]
    image_width = padded_image_width - 2
    
    # prepare output array for unpadded feature maps
    unpadded_num_pixels = image_width ** 2
    unpadded_image = cp.zeros((padded_batch.shape[0], input_channels, unpadded_num_pixels), dtype=cp.float32)

    # remove padding from feature maps (copy data into unpadded array)
    for index, feature_map in enumerate(padded_batch):
        for channel in range(input_channels):
            padded_image_i = image_width + 3
            unpadded_image_j = 0
            while unpadded_image_j < unpadded_num_pixels:
                unpadded_image[index, channel, unpadded_image_j:unpadded_image_j + image_width] = feature_map[channel, padded_image_i:padded_image_i + image_width]
                padded_image_i += padded_image_width
                unpadded_image_j += image_width

    return unpadded_image


def conv_2d(batch, kernels):
    
    # get feature map and kernel sizes
    batch_size = batch.shape[0]
    input_channels = batch.shape[1]
    output_channels = kernels.shape[0]
    
    feature_map_width = int(batch.shape[2] ** .5)
    kernel_width = int(kernels.shape[2] ** .5)
    
    # prepare output array for convolution results
    moves = (feature_map_width - kernel_width + 1) ** 2
    conv_product = cp.zeros((batch_size, output_channels, moves), dtype=cp.float32)

    # perform forward convolution
    for index, feature_map in enumerate(batch):
        print(f"\n\t\tProcessing image {index+1}/{batch_size}")
        
        for o_channel in range(output_channels):
            for i_channel in range(input_channels):
                curr_kernel = kernels[o_channel, i_channel, :]
                
                for move in range(moves):
                    kernel_sum = 0
                    for kernel_index in range(kernels.shape[2]):
                        kernel_sum += curr_kernel[kernel_index] * feature_map[i_channel, (move + ((kernel_index // kernel_width) * feature_map_width) + kernel_index % kernel_width)]
                    conv_product[index, o_channel, move] += kernel_sum
                    
        print(f"\t\tDone.")
    return conv_product

def conv_2d_backpass(batch, deriv_output_maps):
    
    # get dimensions for gradient calculation
    batch_size = batch.shape[0]
    input_channels = batch.shape[1]
    output_channels = deriv_output_maps.shape[1]
    
    feature_map_width = int(batch.shape[2] ** .5)
    output_map_width = int(deriv_output_maps.shape[2] ** .5)
    kernel_width = int(feature_map_width - output_map_width + 1)
    
    # prepare output array for convolution results
    moves = (feature_map_width - kernel_width + 1) ** 2
    conv_product = cp.zeros((batch_size, output_channels, input_channels, kernel_width**2), dtype=cp.float32)

    # perform backprop through conv layer
    for index, feature_map in enumerate(batch):
        print(f"\n\t\tProcessing image {index+1}/{batch_size}")
        
        for o_channel in range(output_channels):
            curr_deriv_map = deriv_output_maps[index, o_channel, :]
            for i_channel in range(input_channels):
                
                for move in range(moves):
                    kernel_sum = 0
                    for kernel_index in range(kernel_width**2):
                        kernel_sum += curr_deriv_map[move] * feature_map[i_channel, (move + ((kernel_index // kernel_width) * feature_map_width) + kernel_index % kernel_width)]
                    conv_product[index, o_channel, i_channel, kernel_index] += kernel_sum
                    
        print(f"\t\tDone.")
    return conv_product

def full_conv_2d(batch, kernels):
    
    # get feature map and kernel sizes
    batch_size = batch.shape[0]
    output_channels = batch.shape[1]
    input_channels = kernels.shape[1]
    
    feature_map_width = int(batch.shape[2] ** .5)
    kernel_width = int(kernels.shape[2] ** .5)
    
    # prepare output array for convolution results
    moves = (feature_map_width - kernel_width + 1) ** 2
    conv_product = cp.zeros((batch_size, input_channels, moves), dtype=cp.float32)

    # perform full convolution through conv layer
    for index, feature_map in enumerate(batch):
        print(f"\n\t\tProcessing image {index+1}/{batch_size}")
        
        for i_channel in range(input_channels):
            for o_channel in range(output_channels):
                curr_kernel = kernels[o_channel, i_channel, :]
                reversed_kernel = curr_kernel[::-1]
                
                for move in range(moves):
                    kernel_sum = 0
                    for kernel_index in range(kernels.shape[2]):
                        kernel_sum += reversed_kernel[kernel_index] * feature_map[o_channel, (move + ((kernel_index // kernel_width) * feature_map_width) + kernel_index % kernel_width)]
                    conv_product[index, i_channel, move] += kernel_sum
                    
        print(f"\t\tDone.")
    return conv_product


def max_pooling(batch):
    
    # get sizes of feature maps and pooling kernel
    batch_size = batch.shape[0]
    input_channels = batch.shape[1]
    
    kernel_width = 2
    image_width = int(batch.shape[2] ** .5)
    moves = (image_width // kernel_width) ** 2

    # prepare output arrays
    pooled_maps = cp.zeros((batch_size, input_channels, moves), dtype=cp.float32)
    pooled_indices = cp.zeros((batch_size, input_channels, moves), dtype=cp.int32)

    # apply max pooling
    for index, feature_map in enumerate(batch):
        for i_channel in range(input_channels):
            curr_kernel = cp.zeros(kernel_width**2, dtype=cp.float32) 
            
            for move in range(0, moves + 2, 2):
                for kernel_index in range(kernel_width**2):
                    curr_kernel[kernel_index] = feature_map[i_channel, (move + ((kernel_index // kernel_width) * image_width) + kernel_index % kernel_width)]
                
                pooled_maps[index, i_channel, move // 2] = cp.max(curr_kernel)
                pooled_indices[index, i_channel, move // 2] = cp.argmax(curr_kernel)

    return pooled_maps, pooled_indices

def unpooling(pooled_maps, pooled_indices, num_pixels):
    
    # get feature map info
    batch_size, channels, num_pooled_pixels = pooled_maps.shape
    unpooled_output = cp.zeros((batch_size, channels, num_pixels), dtype=cp.float32)

    # restore pooled values to original positions
    for image in range(batch_size):
        for channel in range(channels):
            
            for j in range(num_pooled_pixels):
                
                pooled_value = pooled_maps[image, channel, j]
                original_index = pooled_indices[image, channel, j]
                
                unpooled_output[image, channel, original_index] = pooled_value

    return unpooled_output


def forward_prop(hidden_weights, hidden_bias, output_weights, output_bias, batch_size, batch):
    # get layer sizes
    hidden_neurons = hidden_weights.shape[0]
    output_neurons = output_weights.shape[0]

    # prepare output arrays
    pre_act_hidden = cp.zeros((batch_size, hidden_neurons), dtype=cp.float32)
    act_hidden = cp.zeros((batch_size, hidden_neurons), dtype=cp.float32)
    
    pre_act_output = cp.zeros((batch_size, output_neurons), dtype=cp.float32)
    act_output = cp.zeros((batch_size, output_neurons), dtype=cp.float32)

    # forward pass
    for i, image in enumerate(batch):
        
        image_pre_act_hidden = hidden_weights.dot(image) + hidden_bias
        image_act_hidden = ReLU(image_pre_act_hidden)
        
        image_pre_act_output = output_weights.dot(image_act_hidden) + output_bias
        image_act_output = softmax(image_pre_act_output)

        # store results
        pre_act_hidden[i] = image_pre_act_hidden
        act_hidden[i] = image_act_hidden
        
        pre_act_output[i] = image_pre_act_output
        act_output[i] = image_act_output

    return pre_act_hidden, act_hidden, pre_act_output, act_output

def FCL_backward_prop(pre_act_hidden, act_hidden, pre_act_output, act_output, true_labels, hidden_weights, output_weights, batch, batch_size):
    
    # output layer gradients
    deriv_pre_act_output = act_output - true_labels
    deriv_weight_output = (1 / batch_size) * deriv_pre_act_output.T.dot(act_hidden)
    deriv_bias_output = (1 / batch_size) * cp.sum(deriv_pre_act_output, axis=0, keepdims=True)

    # hidden layer gradients
    deriv_act_hidden = deriv_pre_act_output.dot(output_weights)
    deriv_pre_act_hidden = deriv_act_hidden * ReLU_deriv(pre_act_hidden)
    deriv_weight_hidden = (1 / batch_size) * deriv_pre_act_hidden.T.dot(batch)
    deriv_bias_hidden = (1 / batch_size) * cp.sum(deriv_pre_act_hidden, axis=0, keepdims=True)

    # input gradient
    deriv_input = deriv_pre_act_hidden.dot(hidden_weights)

    return deriv_weight_output, deriv_bias_output, deriv_weight_hidden, deriv_bias_hidden, deriv_input


# utility functions ----------------------
def one_hot(Y, output_size):
    one_hot_Y = cp.zeros(shape=(Y.size, output_size), dtype=cp.float32)
    one_hot_Y[cp.arange(Y.size), Y] = 1
    
    return one_hot_Y

def ReLU(pre_act):
    return cp.maximum(pre_act, 0)

def ReLU_deriv(Z):
    return Z > 0

def softmax(pre_act): # output layer activation
    act = cp.exp(pre_act) / cp.sum(cp.exp(pre_act), axis=0)
    return act

# ----------------------------------------


def gradient_descent(num_epochs, batch_size, learning_rate, input_data, labels):

    # prepare batches
    
    num_samples = input_data.shape[0]
    num_batches = num_samples // batch_size
    
    if num_samples % batch_size != 0:
        num_batches += 1
    
    # initialize parameters
    conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias = init_params(
    
        KERNEL_SIZE,
        CONV1_SIZE,
        CONV2_SIZE,
        INPUT_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE,
    )
    
    # train for a specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")

        # shuffle data each epoch
        permutation = cp.random.permutation(num_samples)
        shuffled_images = input_data[permutation]
        shuffled_labels = labels[permutation]
        
        # perform gradient descent over all batches
        for i in range(num_batches):
            
            # get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            curr_batch_size = min(batch_size, num_samples - start_idx)

            batch_images = shuffled_images[start_idx:end_idx] # Current batch of images
            batch_labels = shuffled_labels[start_idx:end_idx] # Current batch of labels
            label_one_hot = one_hot(batch_labels, OUTPUT_SIZE)
            print("----------------------------------------------------------------\n")
            print(f"Processing batch {i+1}/{num_batches} of epoch {epoch+1}")
            
            # ------------ forward pass ------------
            
            # maps = "feature maps"
            
            # forward pass through conv layer 1
            
            print("\tPerforming forward propagation through conv layer 1")
            padded_maps1 = zero_padding(batch_images)
            conv_product1 = conv_2d(padded_maps1, conv1_weights) + conv1_bias
            relu_output1 = ReLU(conv_product1)

            pooled_maps1, pooled_indeces1 = max_pooling(relu_output1)
            
            # forward pass through conv layer 2
            
            print("\tPerforming forward propagation through conv layer 2")
            padded_maps2 = zero_padding(pooled_maps1)
            conv_product2 = conv_2d(padded_maps2, conv2_weights) + conv2_bias
            relu_output2 = ReLU(conv_product2)

            pooled_maps2, pooled_indeces2 = max_pooling(relu_output2)
            
            flattened_maps = cp.reshape(a=pooled_maps2, newshape=(curr_batch_size, INPUT_SIZE))
            
            # forward pass through fully connected layers
            
            print("\tPerforming forward propagation through the fully connected layers")
            pre_act_hidden, act_hidden, pre_act_output, act_output = forward_prop(
                hidden_weights, 
                hidden_bias, 
                output_weights, 
                output_bias,
                curr_batch_size,
                flattened_maps
            )
            
            # calculate sum of losses for the batch
            per_example_loss = -cp.sum(label_one_hot * cp.log(act_output), axis=1)

            # average loss for the batch
            avg_loss = cp.mean(per_example_loss)
            
            print("-----------------------------------\n")
            print(f"\nguesses: {cp.argmax(act_output, axis=1)}")
            print(f"labels: {batch_labels}\n")
            print(f"  Batch {i+1}/{num_batches}, Average Loss: {avg_loss:.4f}")
            print("\n-----------------------------------")
            
            # ------------ backard pass ------------
            
            # backprop through fully connected layers
            
            print("\tPerforming backward propagation through the fully connected layers")
            deriv_weight_output, deriv_bias_output, deriv_weight_hidden, deriv_bias_hidden, deriv_flattened_maps = FCL_backward_prop(
                pre_act_hidden,
                act_hidden,
                pre_act_output,
                act_output,
                label_one_hot,
                hidden_weights,
                output_weights,
                flattened_maps,
                batch_size    
            )

            # unflatten the gradients to match the pooled feature map shape
            deriv_pooled_maps2 = cp.reshape(deriv_flattened_maps, pooled_maps2.shape)
            
            # backprop through conv layer 2
            
            deriv_unpool_maps2 = unpooling(deriv_pooled_maps2, pooled_indeces2, relu_output2.shape[2])
            deriv_relu_output2 = deriv_unpool_maps2 * ReLU_deriv(conv_product2)

            print("\tBackpropagating through Conv layer 2")
            kernel_gradients2 = conv_2d_backpass(padded_maps2, deriv_relu_output2)
            deriv_conv2_bias = cp.sum(deriv_relu_output2, axis=(0, 2), keepdims=True)
            padded_deriv_relu_output2 = zero_padding(deriv_relu_output2)
            feature_map_gradients2 = full_conv_2d(padded_deriv_relu_output2, conv2_weights)
            
            # backprop through conv layer 1

            unpadded_feature_gradients = zero_unpadding(feature_map_gradients2)
            deriv_unpool_maps1 = unpooling(unpadded_feature_gradients, pooled_indeces1, relu_output1.shape[2])

            deriv_relu_output1 = deriv_unpool_maps1 * ReLU_deriv(conv_product1)

            print("\tBackpropagating through Conv layer 1")
            kernel_gradients1 = conv_2d_backpass(padded_maps1, deriv_relu_output1)
            deriv_conv1_bias = cp.sum(deriv_relu_output1, axis=(0, 2), keepdims=True)
            
            # ------------ update parameters ------------
            
            print("\tUpdating parameters")
            conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias = update_params(
            
            conv1_weights,conv1_bias,conv2_weights,conv2_bias,hidden_weights,hidden_bias,output_weights,output_bias, # params
            
            deriv_weight_hidden,deriv_bias_hidden,deriv_weight_output,deriv_bias_output,kernel_gradients2,deriv_conv2_bias,         
            kernel_gradients1,deriv_conv1_bias, # gradients
               
            learning_rate, batch_size #self explanatatory
            )
            
            
            # clear gradients for memory efficiency
            deriv_weight_hidden.fill(0)
            deriv_bias_hidden.fill(0)
            deriv_weight_output.fill(0)
            deriv_bias_output.fill(0)
            kernel_gradients2.fill(0)
            deriv_conv2_bias.fill(0)
            kernel_gradients1.fill(0)
            deriv_conv1_bias.fill(0)
            
            print("----------------------------------------------------------------\n")
        print(f"Epoch {epoch+1} completed.")
    return conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias
    
def evalute_model(conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias, input_data, labels, epoch):
    
    num_samples = input_data.shape[0]
    batch_size = 100
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_batches += 1
    
    correct_predictions = 0
    
    for i in range(num_batches):
        
        # get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        curr_batch_size = min(batch_size, num_samples - start_idx)

        batch_images = input_data[start_idx:end_idx] # Current batch of images
        batch_labels = labels[start_idx:end_idx] # Current batch of labels
        label_one_hot = one_hot(batch_labels, OUTPUT_SIZE)
        
        # ------------ forward pass ------------
        
        padded_images1 = zero_padding(batch_images)
        conv_product1 = conv_2d(padded_images1, conv1_weights) + conv1_bias
        relu_output1 = ReLU(conv_product1)

        pooled_images1, _ = max_pooling(relu_output1)
        
        padded_images2 = zero_padding(pooled_images1)

        conv_product2 = conv_2d(padded_images2, conv2_weights) + conv2_bias

        relu_output2 = ReLU(conv_product2)

        pooled_images2, _ = max_pooling(relu_output2)
        
        flattened_images = cp.reshape(a=pooled_images2, newshape=(curr_batch_size, INPUT_SIZE))
        
        _, _, _, act_output = forward_prop(
            hidden_weights, 
            hidden_bias, 
            output_weights, 
            output_bias,
            curr_batch_size,
            flattened_images
        )
        
        predictions = cp.argmax(act_output, axis=1)
        correct_predictions += cp.sum(predictions == batch_labels)
    
    accuracy = correct_predictions / num_samples
    print(f"Model accuracy for dev set: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    
    # load the dataset
    data = cp.loadtxt('MNIST/mnist_train.csv', delimiter=',', skiprows=1, dtype=cp.float32)
    m_num_examples, num_features = data.shape
    cp.random.shuffle(data) # shuffle before splitting into dev and training sets
    
    # !!! would need to implement extracting the different channels. This implementation assumes grayscale images only !!!
    
    # extract dev set
    dev_data = data[0:1000]
    dev_labels = dev_data[:, 0].astype(cp.int32)        # dev labels (shape 1000,)
    dev_features = dev_data[:, 1:num_features]          # dev features (shape 1000, 784)
    dev_features = dev_features / 255.                  # normalize dev pixel values

    # extract training set
    train_data = data[1000:m_num_examples]
    train_labels = train_data[:, 0].astype(cp.int32)    # training labels (shape 59000,)
    train_features = train_data[:, 1:num_features]      # training features (shape 59000, 784)
    train_features = train_features / 255.              # normalize training pixel values

    # add the 'channels' dimension (grayscale images)
    dev_features = cp.expand_dims(dev_features, axis=1)
    train_features = cp.expand_dims(train_features, axis=1)


    # hyper-parameters
    NUM_EPOCHS = 200 # arbitrary number like 10, 50, 100, 200, etc.
    BATCH_SIZE = 64 # 32, 64, 128, 256, etc. (powers of 2 >= 32)
    LEARNING_RATE = 0.001 # arbritrary small value like 0.001, 0.01, 0.1, etc.
    KERNEL_SIZE = 9 # 3x3 kernels = 9, 5x5 kernels = 25, etc.
    CONV1_SIZE = 32 # number of kernels in conv layer 1
    CONV2_SIZE = 64 # number of kernels in conv layer 2
    INPUT_SIZE = (int(train_features.shape[2] ** .5)//4) ** 2 * 64
    HIDDEN_SIZE = 256 # number of neurons in hidden layer
    OUTPUT_SIZE = 10 # number of output classes (Example: digits 0-9)
    
    print("\nTraining the model...\n")
    
    conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias = gradient_descent(
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, train_features, train_labels)
    
    print("\nModel trained.\n")
    
    evalute_model(conv1_weights, conv1_bias, conv2_weights, conv2_bias, hidden_weights, hidden_bias, output_weights, output_bias, dev_features, dev_labels)
    