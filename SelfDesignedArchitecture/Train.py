import IO
import PatternManipulator
import Model

Margin = PatternManipulator.GetMargin([9, 1, 5])
Step = 12

HiFreq = IO.LoadRawBinaryGrayscaleSequence('HiFreq.bin', 512, 512, 4)
Base = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 4)
Target = IO.LoadRawBinaryGrayscaleSequence('HiRes.bin', 512, 512, 4)
HiFreq = PatternManipulator.ExtractGrayscalePatterns(HiFreq, 33, Step)
Base = PatternManipulator.ExtractGrayscalePatterns(Base, 33, Step, Margin)
Target = PatternManipulator.ExtractGrayscalePatterns(Target, 33, Step, Margin)
HiFreq = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(HiFreq)
Base = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Base)
Target = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Target)
Model = Model.GetModel()
Model.fit([HiFreq, Base], Target, epochs=10000)
Model.save('Model.h5')