# Inception Transfer Learning
# Gerrit van Rensburg
# 2022-03-29
# Followed instructions from Tekan Irla - https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b


# Import that inception model
from re import X
from tensorflow.keras.application.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape = (150, 150, 3),   # Shape of the images
                include_top = False,    # Leave out the last fully connected layer, specific for ImageNet
                weights = 'imagenet')

# make all layers non-trainable
# retain some lower layers to increase performace can lead to overfitting

for layer in pre_trained_model.layers:
    layer.trainable = False

# RMSprop with a learning rate of 0.0001 - can also experiment with Adam and Adagran optimizers

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification
x = layers.Dense (1, activation='sigmoid')(x)

model = Model( pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
            loss = 'binary_crossentropy',
            metrics = ['acc'])

        
