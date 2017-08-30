import IO
import PatternManipulator
import Model

PatternSize = 159
Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,9])
Step = 12

HiFreq = IO.LoadRawBinaryGrayscaleSequence('HiFreq.bin', 512, 512, 1)
Base = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 1)
Target = IO.LoadRawBinaryGrayscaleSequence('HiRes.bin', 512, 512, 1)
HiFreq = PatternManipulator.ExtractGrayscalePatterns(HiFreq, PatternSize, Step)
Base = PatternManipulator.ExtractGrayscalePatterns(Base, PatternSize, Step, Margin)
Target = PatternManipulator.ExtractGrayscalePatterns(Target, PatternSize, Step, Margin)
HiFreq = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(HiFreq)
Base = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Base)
Target = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Target)
Model = Model.GetModel()
Model.fit([HiFreq, Base], Target, batch_size=8, epochs=300)
Model.save('Model.h5')