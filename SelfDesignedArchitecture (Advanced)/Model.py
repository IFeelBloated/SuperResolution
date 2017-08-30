from keras import models, layers, optimizers
import PatternManipulator

def GetModel():
    PatternSize = 159
    Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,9])
    HiFreqGuess = layers.Input(shape=(PatternSize, PatternSize, 1))
    LowResBase = layers.Input(shape=(PatternSize - 2 * Margin, PatternSize - 2 * Margin, 1))
    
    def VGGConv(Input, FilterCount, MergeLayer=None):
        Conv = layers.Conv2D(FilterCount, (3, 3), activation=None, padding='valid')(Input)
        if MergeLayer is not None:
           Conv = layers.add([Conv, MergeLayer])
        Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2])(Conv)
        return Conv

    Conv1 = VGGConv(HiFreqGuess, 64)
    Conv1 = VGGConv(Conv1, 64)

    Conv2 = VGGConv(Conv1, 128)
    Conv2 = VGGConv(Conv2, 128)

    Conv3 = VGGConv(Conv2, 256)
    Conv3 = VGGConv(Conv3, 256)
    Conv3 = VGGConv(Conv3, 256)
    Conv3 = VGGConv(Conv3, 256)

    Conv4 = VGGConv(Conv3, 512)
    Conv4 = VGGConv(Conv4, 512)
    Conv4 = VGGConv(Conv4, 512)
    Conv4 = VGGConv(Conv4, 512)

    Conv5 = VGGConv(Conv4, 512)
    Conv5 = VGGConv(Conv5, 512)
    Conv5 = VGGConv(Conv5, 512)
    Conv5 = VGGConv(Conv5, 512)

    Map5 = VGGConv(Conv5, 512)
    Map5 = VGGConv(Map5, 512)
    Map5 = VGGConv(Map5, 512)
    Map5 = VGGConv(Map5, 512, layers.convolutional.Cropping2D(4)(Conv5))

    Map4 = VGGConv(Map5, 512)
    Map4 = VGGConv(Map4, 512)
    Map4 = VGGConv(Map4, 512)
    Map4 = VGGConv(Map4, 512, layers.convolutional.Cropping2D(12)(Conv4))

    Map3 = VGGConv(Map4, 256)
    Map3 = VGGConv(Map3, 256)
    Map3 = VGGConv(Map3, 256)
    Map3 = VGGConv(Map3, 256, layers.convolutional.Cropping2D(20)(Conv3))

    Map2 = VGGConv(Map3, 128)
    Map2 = VGGConv(Map2, 128, layers.convolutional.Cropping2D(26)(Conv2))

    Map1 = VGGConv(Map2, 64)
    Map1 = VGGConv(Map1, 64, layers.convolutional.Cropping2D(30)(Conv1))

    Ensemble = layers.Conv2D(1, (9, 9), activation=None, use_bias=False, padding='valid')(Map1)
    HighResolutionPatterns = layers.add([LowResBase, Ensemble])
    
    Model = models.Model(inputs=[HiFreqGuess, LowResBase], outputs=HighResolutionPatterns)
    Model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4, decay=5e-4))
    return Model