from keras import models, layers, optimizers

def GetModel():
    Model = models.Sequential()
    Model.add(layers.Conv2D(64, (9, 9), activation='relu', input_shape=(1, 33, 33)))
    Model.add(layers.Conv2D(32, (1, 1), activation='relu'))
    Model.add(layers.Conv2D(1, (5, 5), activation=None))
    Model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001))
    return Model
