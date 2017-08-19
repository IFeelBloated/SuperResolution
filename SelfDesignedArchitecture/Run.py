import IO
import PatternManipulator
import keras
import numpy

PatternSize = 33
Margin = PatternManipulator.GetMargin([9, 1, 5])

def ReconstructGrayscaleSequence(HiFreqSequence, BaseSequence, PatternSize, Margin):
    FrameCount = BaseSequence.shape[0]
    Height = BaseSequence.shape[1]
    Width = BaseSequence.shape[2]
    ReconstructedSequence = []
    def FetchSingleFrameFromGrayscaleSequence(Sequence, Index):
        Frame = numpy.zeros((1, Height, Width))
        Frame[0,:,:] = Sequence[Index,:,:]
        return Frame
    def ReconstructSingleFrame(HiFreq, Base):
        Patterns = PatternManipulator.ExtractGrayscalePatterns(HiFreq, PatternSize, PatternSize - 2 * Margin)
        Patterns = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Patterns)
        Reference = PatternManipulator.ExtractGrayscalePatterns(Base, PatternSize, PatternSize - 2 * Margin, Margin)
        Reference = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Reference)
        ReconstructedPatterns = Model.predict([Patterns, Reference])
        ReconstructedFrame = PatternManipulator.AssembleGrayscalePatterns(ReconstructedPatterns[:,:,:,0], Width, Height, Margin)
        return ReconstructedFrame
    def FrameListToArray():
        Array = numpy.zeros(BaseSequence.shape)
        for i in range(FrameCount):
            Array[i,:,:] = ReconstructedSequence[i]
        return Array
    for i in range(FrameCount):
        HiFreq = FetchSingleFrameFromGrayscaleSequence(HiFreqSequence, i)
        Base = FetchSingleFrameFromGrayscaleSequence(BaseSequence, i)
        ReconstructedSequence += [ReconstructSingleFrame(HiFreq, Base)]
    return FrameListToArray()
    
Model = keras.models.load_model('Model.h5')
Base = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 4)
HiFreq = IO.LoadRawBinaryGrayscaleSequence('HiFreq.bin', 512, 512, 4)
Reconstructed = ReconstructGrayscaleSequence(HiFreq, Base, PatternSize, Margin)
IO.ExportRawBinaryGrayscaleSequence(Reconstructed, 'Reconstructed.bin')