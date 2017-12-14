# @pierrotechnique
# -*- coding: utf-8 -*-

import pyo
import time

# Server object with default arguments
# Server(sr=44100,nchnls=2,buffersize=256,duplex=1,...)
# duplex sets input-output mode: 0 output only, 1 both
s = pyo.Server()

# Boot server (init audio + MIDI streams): audio + MIDI configs before this
# See doc for complete list of methods to be called before boot
s.boot()

# Start server: activates server processing/audio callback loop
s.start()

# Signal processing chain here

s.amp = 0.1 # Amp gain to -20 dB

a1 = pyo.Sine(220.).out(0) # Plays a sine wave and sends it out on ch1/L (0)
b1 = pyo.Sine(440.).out(1) # Plays a sine wave and sends it out on ch2/R (1)
# Sine(freq,phase)
# out(chnl)

# Processes called on the same object are parallelized!
hr_a1 = pyo.Harmonizer(a1).out(0)
hr_b1 = pyo.Harmonizer(b1).out(1)
ch_a1 = pyo.Chorus(a1).out(0)
ch_b1 = pyo.Chorus(b1).out(1)

a2 = hr_a1 + ch_a1
b2 = hr_b1 + ch_b1

sh_a2 = pyo.FreqShift(a2).out(0)
sh_b2 = pyo.FreqShift(b2).out(1)

time.sleep(4.4)

# Stop audio callback loop
s.stop()