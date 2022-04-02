# Inception Transfer Learning
# Gerrit van Rensburg
# 2022-03-29
# Followed instructions from Tekan Irla - https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b


# Import that inception model
from re import X
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

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
x = tf.keras.layers.Flatten()(pre_trained_model.output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense (1, activation='sigmoid')(x)

model = tf.keras.models.Model( pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
            loss = 'binary_crossentropy',
            metrics = ['acc'])

        


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                rotation_range = 40,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(resale = 1.0/255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (150,150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size = 20,
                                                        class_mode = 'binary',
                                                        target_size = (150,150))

# For 2000 training images (batches of 20) and 1000 validation images (batches of 20)

callbacks = myCallback()
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks]
)