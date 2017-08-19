from keras import models, layers, optimizers

def GetModel():
    HiFreqGuess = layers.Input(shape=(33, 33, 1))
    LowResBase = layers.Input(shape=(21, 21, 1))
    Layer = layers.Conv2D(64, (9, 9), activation=None)(HiFreqGuess)
    Layer = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Layer)
    Layer = layers.Conv2D(32, (1, 1), activation=None)(Layer)
    Layer = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Layer)
    Layer = layers.Conv2D(1, (5, 5), activation=None)(Layer)
    HighResolutionPatterns = layers.add([LowResBase, Layer])
    Model = models.Model(inputs=[HiFreqGuess, LowResBase], outputs=HighResolutionPatterns)
    Model.compile(loss='mse', optimizer=optimizers.Adam())
    return Model
