from keras import models, layers, optimizers
import PatternManipulator

def GetModel():
    PatternSize = 97
    Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,1,1,1,1,9])
    HiFreqGuess = layers.Input(shape=(PatternSize, PatternSize, 1))
    LowResBase = layers.Input(shape=(PatternSize - 2 * Margin, PatternSize - 2 * Margin, 1))
    
    Conv1 = layers.Conv2D(64, (3, 3), activation=None, padding='valid')(HiFreqGuess)
    Conv1 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv1)
    Conv1 = layers.Conv2D(64, (3, 3), activation=None, padding='valid')(Conv1)
    Conv1 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv1)

    Conv2 = layers.Conv2D(128, (3, 3), activation=None, padding='valid')(Conv1)
    Conv2 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv2)
    Conv2 = layers.Conv2D(128, (3, 3), activation=None, padding='valid')(Conv2)
    Conv2 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv2)

    Conv3 = layers.Conv2D(256, (3, 3), activation=None, padding='valid')(Conv2)
    Conv3 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv3)
    Conv3 = layers.Conv2D(256, (3, 3), activation=None, padding='valid')(Conv3)
    Conv3 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv3)

    Conv4 = layers.Conv2D(512, (3, 3), activation=None, padding='valid')(Conv3)
    Conv4 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv4)
    Conv4 = layers.Conv2D(512, (3, 3), activation=None, padding='valid')(Conv4)
    Conv4 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv4)

    Map1 = layers.Conv2D(512, (1, 1), activation=None)(Conv4)
    Map1 = layers.add([Conv4, Map1])
    Map1 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Map1)

    Map2 = layers.Conv2D(256, (1, 1), activation=None)(Map1)
    Conv3 = layers.convolutional.Cropping2D(2)(Conv3)
    Map2 = layers.add([Conv3, Map2])
    Map2 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Map2)

    Map3 = layers.Conv2D(128, (1, 1), activation=None)(Map2)
    Conv2 = layers.convolutional.Cropping2D(4)(Conv2)
    Map3 = layers.add([Conv2, Map3])
    Map3 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Map3)

    Map4 = layers.Conv2D(64, (1, 1), activation=None)(Map3)
    Conv1 = layers.convolutional.Cropping2D(6)(Conv1)
    Map4 = layers.add([Conv1, Map4])
    Map4 = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Map4)

    Ensemble = layers.Conv2D(1, (9, 9), activation=None, padding='valid')(Map4)
    HighResolutionPatterns = layers.add([LowResBase, Ensemble])
    
    Model = models.Model(inputs=[HiFreqGuess, LowResBase], outputs=HighResolutionPatterns)
    Model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4, decay=5e-4))
    return Model