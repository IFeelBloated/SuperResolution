import IO
import PatternManipulator
import keras
import numpy

PatternSize = 33
Margin = PatternManipulator.GetMargin([9, 1, 5])

def ReconstructGrayscaleSequence(Sequence, PatternSize, Margin):
    FrameCount = Sequence.shape[0]
    Height = Sequence.shape[1]
    Width = Sequence.shape[2]
    ReconstructedSequence = []
    def FetchSingleFrameFromGrayscaleSequence(Index):
        Frame = numpy.zeros((1, Height, Width))
        Frame[0,:,:] = Sequence[Index,:,:]
        return Frame
    def ReconstructSingleFrame(Frame):
        Patterns = PatternManipulator.ExtractGrayscalePatterns(Frame, PatternSize, PatternSize - 2 * Margin)
        Patterns = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Patterns)
        ReconstructedPatterns = Model.predict(Patterns)
        ReconstructedFrame = PatternManipulator.AssembleGrayscalePatterns(ReconstructedPatterns[:,:,:,0], Width, Height, Margin)
        return ReconstructedFrame
    def FrameListToArray():
        Array = numpy.zeros(Sequence.shape)
        for i in range(FrameCount):
            Array[i,:,:] = ReconstructedSequence[i]
        return Array
    for i in range(FrameCount):
        ReconstructedSequence += [ReconstructSingleFrame(FetchSingleFrameFromGrayscaleSequence(i))]
    return FrameListToArray()
    
Model = keras.models.load_model('Model.h5')
Input = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 4)
Reconstructed = ReconstructGrayscaleSequence(Input, PatternSize, Margin)
IO.ExportRawBinaryGrayscaleSequence(Reconstructed, 'Reconstructed.bin')