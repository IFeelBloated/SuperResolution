import vapoursynth as vs
core = vs.get_core()
clip = core.raws.Source('HiRes.bin', 512, 512, src_fmt="GRAYS")
clip = core.fmtc.resample(clip, 256, 256, kernel="sinc", taps=128, fulls=True, fulld=True)
clip = core.fmtc.resample(clip, 512, 512, kernel="sinc", taps=128, fulls=True, fulld=True)
clip.set_output()