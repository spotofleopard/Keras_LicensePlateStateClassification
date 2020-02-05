from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D,Reshape,LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import os,numpy as np

batch_size=128
training_set_folder=os.path.expanduser('~/dataset/LPR/plates_128x64/train')
val_set_folder=os.path.expanduser('~/dataset/LPR/plates_128x64/test')
num_samples=cpt = sum([len(files) for r, d, files in os.walk(training_set_folder)])

# dataset information
input_shape = (64, 128, 3)
classes= ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
num_classes = len(classes)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        brightness_range=(0.8,1.2))

train_generator = train_datagen.flow_from_directory(
        training_set_folder,
        target_size=(64, 128),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

val_datagen=ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_directory(
        val_set_folder,
        target_size=(64, 128),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical')

base_model = ResNet50V2(input_shape=input_shape,weights='imagenet',include_top=False)
top_model = tf.keras.models.Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu',activity_regularizer=l2(0.0003)))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))
model = Model(inputs = base_model.input, outputs = top_model(base_model.output))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

tensorBoard=TensorBoard()
checkPoint=ModelCheckpoint(filepath='models/state_ep{epoch}.h5',
            save_best_only=True,monitor='val_loss',)
model.fit(train_generator,
          steps_per_epoch=num_samples//batch_size,
          epochs=30,
          verbose=1,
          validation_data=validation_generator,
          callbacks=[tensorBoard,checkPoint])
