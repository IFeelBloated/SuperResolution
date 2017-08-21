import vapoursynth as vs
core = vs.get_core()
clip = core.raws.Source('HiRes.bin', 512, 512, src_fmt="GRAYS")
clip = core.fmtc.resample(clip, 256, 256, kernel="sinc", taps=128, fulls=True, fulld=True)
lowres = core.fmtc.resample(clip, 128, 128, kernel="sinc", taps=128, fulls=True, fulld=True)
lowres = core.fmtc.resample(lowres, 512, 512, kernel="sinc", taps=128, fulls=True, fulld=True)
clip = core.fmtc.resample(clip, 512, 512, kernel="sinc", taps=128, fulls=True, fulld=True)
clip = core.std.MakeDiff(clip,lowres)
clip.set_output()