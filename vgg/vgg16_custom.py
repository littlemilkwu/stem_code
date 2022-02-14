from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

input_shape = (224, 224, 3)

model = Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding="same"))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1000, activation='softmax'))


print(model.summary())
