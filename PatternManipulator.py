import numpy

def GetMargin(FilterSizes):
    Margin = 0
    for x in FilterSizes:
        Margin += (x - 1) // 2
    return Margin

def ExtractGrayscalePatterns(Sequence, Size, Step, Margin=0):
    ImageCount = Sequence.shape[0]
    PatternList = []
    def SliceGrayscaleImage(Image):
        Height = Image.shape[0]
        Width = Image.shape[1]
        PatternList = []
        for i in range(0, Height - Size + 1, Step):
            for j in range(0, Width - Size + 1, Step):
                PatternList += [Image[j + Margin:j - Margin + Size, i + Margin:i - Margin + Size]]
        return PatternList
    def PatternListTo3DArray():
        PatternCount = len(PatternList)
        Array = numpy.zeros((PatternCount,) + PatternList[0].shape)
        for i in range(PatternCount):
            Array[i,:,:] = PatternList[i]
        return Array
    for i in range(ImageCount):
        PatternList += SliceGrayscaleImage(Sequence[i,:,:])
    return PatternListTo3DArray()

def AssembleGrayscalePatterns(Patterns, Width, Height, Margin):
    PatternSize = Patterns.shape[1]
    Canvas = numpy.zeros((Height, Width))
    HorizontalPatternCount = (Width - 2 * Margin) // PatternSize
    for i in range(0, Height - PatternSize - 2 * Margin + 1, PatternSize):
        for j in range(0, Width - PatternSize - 2 * Margin + 1, PatternSize):
            CorrespondingIndex = (i * HorizontalPatternCount + j) // PatternSize
            Canvas[j + Margin:j + Margin + PatternSize, i + Margin:i + Margin + PatternSize] = Patterns[CorrespondingIndex,:,:]
    return Canvas

def GenerateExtraDimensionForGrayscaleSequence(Sequence):
    Array = numpy.zeros(Sequence.shape + (1,))
    Array[:,:,:,0] = Sequence
    return Array