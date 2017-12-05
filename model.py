### Build the model and train
# Based on NVIDIA model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Conv2D, Cropping2D
from keras.callbacks import EarlyStopping

row, col, ch = 160, 320, 3  # camera format from car

model = Sequential()
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(row,col,ch))) # crop 60 pixels from top, 20 pixels from bottom
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(80, 320, 3),
        output_shape=(80, 320, 3))) # normalize layer
model.add(Conv2D(24, 5, strides=(2, 2), padding="valid"))
model.add(ELU())
model.add(Conv2D(36, 5, strides=(2, 2), padding="valid"))
model.add(ELU())
model.add(Conv2D(48, 5, strides=(2, 2), padding="valid"))
model.add(ELU())
model.add(Conv2D(64, 3, strides=(1, 1), padding="valid"))
model.add(ELU())
model.add(Conv2D(64, 3, strides=(1, 1), padding="valid"))
model.add(Flatten())
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(100))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary() # summary of model

early_stopping = EarlyStopping(monitor='val_loss', patience=2) # stop training early if validation loss starts increasing
model.compile(optimizer="adam", loss="mse")
model.fit(X_train,y_train, validation_split=0.1, shuffle=True, epochs=10, callbacks=[early_stopping])
    
model.save('model.h5')
print('model saved')
