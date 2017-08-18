import numpy

def LoadRawBinaryGrayscaleSequence(Address, Width, Height, Length, PixelType='float32'):
    Sequence = numpy.memmap(Address, dtype=PixelType, mode='r', shape=(Length, Height, Width))
    Sequence = numpy.array(Sequence, dtype='float64')
    return Sequence

def ExportRawBinaryGrayscaleSequence(Data, Address, PixelType='float32'):
    Sequence = numpy.memmap(Address, dtype=PixelType, mode='w+', shape=Data.shape)
    Sequence[:] = numpy.array(Data, dtype=PixelType)[:]
    del Sequence