import IO
import PatternManipulator
import Model

Margin = PatternManipulator.GetMargin([9, 1, 5])
Step = 12

Input = IO.LoadRawBinaryGrayscaleSequence('LowRes.bin', 512, 512, 4)
Target = IO.LoadRawBinaryGrayscaleSequence('HiRes.bin', 512, 512, 4)
Input = PatternManipulator.ExtractGrayscalePatterns(Input, 33, Step)
Target = PatternManipulator.ExtractGrayscalePatterns(Target, 33, Step, Margin)
Input = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Input)
Target = PatternManipulator.GenerateExtraDimensionForGrayscaleSequence(Target)
Model = Model.GetModel()
Model.fit(Input, Target, epochs=10000)
Model.save('Model.h5')