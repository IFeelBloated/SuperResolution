from keras import models, layers, optimizers
import PatternManipulator

def GetModel():
    PatternSize = 159
    Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,9])
    HiFreqGuess = layers.Input(shape=(PatternSize, PatternSize, 1), name='HiFreqGuess')
    LowResBase = layers.Input(shape=(PatternSize - 2 * Margin, PatternSize - 2 * Margin, 1), name='LowResBase')
    
    def VGGConv(Input, FilterCount, Tag, MergeLayer=None):
        Conv = layers.Conv2D(FilterCount, (3, 3), activation=None, padding='valid', name='Conv' + Tag)(Input)
        if MergeLayer is not None:
           Conv = layers.add([Conv, MergeLayer])
        Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + Tag)(Conv)
        return Conv

    Conv1 = VGGConv(HiFreqGuess, 64, 'AE1')
    Conv1 = VGGConv(Conv1, 64, 'AE2')

    Conv2 = VGGConv(Conv1, 128, 'BE1')
    Conv2 = VGGConv(Conv2, 128, 'BE2')

    Conv3 = VGGConv(Conv2, 256, 'CE1')
    Conv3 = VGGConv(Conv3, 256, 'CE2')
    Conv3 = VGGConv(Conv3, 256, 'CE3')
    Conv3 = VGGConv(Conv3, 256, 'CE4')

    Conv4 = VGGConv(Conv3, 512, 'DE1')
    Conv4 = VGGConv(Conv4, 512, 'DE2')
    Conv4 = VGGConv(Conv4, 512, 'DE3')
    Conv4 = VGGConv(Conv4, 512, 'DE4')

    Conv5 = VGGConv(Conv4, 512, 'EE1')
    Conv5 = VGGConv(Conv5, 512, 'EE2')
    Conv5 = VGGConv(Conv5, 512, 'EE3')
    Conv5 = VGGConv(Conv5, 512, 'EE4')

    Map5 = VGGConv(Conv5, 512, 'ED4')
    Map5 = VGGConv(Map5, 512, 'ED3')
    Map5 = VGGConv(Map5, 512, 'ED2')
    Map5 = VGGConv(Map5, 512, 'ED1', layers.convolutional.Cropping2D(4)(Conv5))

    Map4 = VGGConv(Map5, 512, 'DD4')
    Map4 = VGGConv(Map4, 512, 'DD3')
    Map4 = VGGConv(Map4, 512, 'DD2')
    Map4 = VGGConv(Map4, 512, 'DD1', layers.convolutional.Cropping2D(12)(Conv4))

    Map3 = VGGConv(Map4, 256, 'CD4')
    Map3 = VGGConv(Map3, 256, 'CD3')
    Map3 = VGGConv(Map3, 256, 'CD2')
    Map3 = VGGConv(Map3, 256, 'CD1', layers.convolutional.Cropping2D(20)(Conv3))

    Map2 = VGGConv(Map3, 128, 'BD2')
    Map2 = VGGConv(Map2, 128, 'BD1', layers.convolutional.Cropping2D(26)(Conv2))

    Map1 = VGGConv(Map2, 64, 'AD2')
    Map1 = VGGConv(Map1, 64, 'AD1', layers.convolutional.Cropping2D(30)(Conv1))

    Ensemble = layers.Conv2D(1, (9, 9), activation=None, use_bias=False, padding='valid', name='Ensemble')(Map1)
    HighResolutionPatterns = layers.add([LowResBase, Ensemble])
    
    Model = models.Model(inputs=[HiFreqGuess, LowResBase], outputs=HighResolutionPatterns)
    Model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4, decay=5e-4))
    return Model