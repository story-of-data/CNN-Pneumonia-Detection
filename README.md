
# Pneumonia Detection using Chest X-ray Images

This repository contains a Jupyter notebook that demonstrates the use of deep learning for detecting pneumonia using chest X-ray images.

## Data Description

- The dataset is sourced from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download).
- It's organized into training, testing, and validation sets.
- The dataset contains images categorized into "PNEUMONIA" and "NORMAL".
- A pandas DataFrame, `df`, is used to represent the labels for these images with "0" indicating "NORMAL" and "1" indicating "PNEUMONIA".

## Model Description

- The model used is VGG16, a pre-trained model on ImageNet.
- Several layers of the VGG16 model are frozen to prevent them from updating during training.
- The output of the VGG16 model is passed through a GlobalAveragePooling layer, a dense layer with 128 neurons (ReLU activation), and finally, a classification layer with 1 neuron (sigmoid activation).
- The learning rate for the model training is scheduled with an exponential decay.
- The model is compiled using the RMSprop optimizer and binary cross-entropy as the loss function.

## Graphs/Figures

- A count plot is present in the notebook which visualizes the distribution of the "PNEUMONIA" and "NORMAL" classes.
- Two graphs are plotted to compare training and validation accuracy, as well as training and validation loss.

## Other Details

- Various paths and image loading techniques are mentioned.
- The model undergoes two phases of training: one with frozen VGG16 layers and another after unfreezing them for fine-tuning.
- The model's performance is evaluated on a test set.

## Note

For more details, please refer to the Jupyter notebook provided in this repository.

## Mathematical Details of the Model

### Convolutional Layer
The convolutional layer is responsible for detecting local features in the input using filters or kernels. The output of this layer is given by:
\[ O = 	ext{ReLU}(I * K + b) \]
Where:
- \( I \) is the input feature map.
- \( K \) is the kernel or filter.
- \( b \) is the bias.
- \( * \) represents the convolution operation.
- \( 	ext{ReLU} \) is the Rectified Linear Unit activation function.

### Pooling Layer
The pooling layer helps in downsampling the feature map. Commonly used pooling is max pooling, which takes the maximum value from a group of values in the feature map.

### Fully Connected Layer
The fully connected layer is given by:
\[ O = 	ext{ReLU}(W 	imes I + b) \]
Where:
- \( W \) is the weight matrix.
- \( I \) is the input from the previous layer.
- \( b \) is the bias.

### Final Classification Layer
For binary classification, the output probability is given by:
\[ P(y=1|I) = rac{1}{1 + e^{-z}} \]
Where:
- \( z = W 	imes I + b \)
- \( P(y=1|I) \) is the probability of the positive class given the input \( I \).

### Loss Function (Binary Cross-Entropy)
The loss function used in this model is binary cross-entropy, given by:
\[ L(y, \hat{y}) = -\left( y \log(\hat{y}) + (1-y) \log(1-\hat{y}) ight) \]
Where:
- \( y \) is the true label.
- \( \hat{y} \) is the predicted probability.
