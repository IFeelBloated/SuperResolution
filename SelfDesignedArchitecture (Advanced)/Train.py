import IO
import PatternManipulator
import Model

PatternSize = 159
Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,9])
Step = 12
batch_size = 8
epochs = 300

HiFreq = IO.LoadRawBinaryGrayscaleSequence('HiFreq.bin', 512, 512, 1)
Base = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 1)
Target = IO.LoadRawBinaryGrayscaleSequence('HiRes.bin', 512, 512, 1)
HiFreq = PatternManipulator.ExtractGrayscalePatterns(HiFreq, PatternSize, Step)
Base = PatternManipulator.ExtractGrayscalePatterns(Base, PatternSize, Step, Margin)
Target = PatternManipulator.ExtractGrayscalePatterns(Target, PatternSize, Step, Margin)
HiFreq = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(HiFreq)
Base = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Base)
Target = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Target)

BlockA = Model.GetBlockA()
BlockA.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
BlockA.save_weights('BlockA.h5')

BlockB = Model.GetBlockB()
BlockB.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
BlockB.save_weights('BlockB.h5')

BlockC = Model.GetBlockC()
BlockC.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
BlockC.save_weights('BlockC.h5')

BlockD = Model.GetBlockD()
BlockD.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
BlockD.save_weights('BlockD.h5')

BlockE = Model.GetBlockE()
BlockE.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
BlockE.save_weights('BlockE.h5')

Model = Model.GetModel()
Model.fit([HiFreq, Base], Target, batch_size = batch_size, epochs = epochs)
Model.save('Model.h5')