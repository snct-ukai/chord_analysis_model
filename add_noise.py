from asyncore import write
import librosa
import csv
import os
import numpy as np
import soundfile as sf
import pathlib

filename : str = "./datalist.csv"
base = os.path.dirname(os.path.abspath(__file__))
file = os.path.normpath(os.path.join(base, filename))
with open(file, "r") as f:
  reader = csv.reader(f)
  list = [row for row in reader]
  inst_l = list[0]
  tone_l = list[1]
  chord_l = list[2]
  group = list[3]

  for inst in inst_l:
    for tone in tone_l:
      for chord in chord_l:
        basepath = tone + "/" + inst + "/" + chord + ".wav"
        path = os.path.normpath(os.path.join(base, group[0] + "/" + basepath))

        data, sr = librosa.load(path)
        wn = np.random.randn(len(data))
        data += 0.005*wn
        
        writepath = os.path.normpath(os.path.join(base, group[1] + "/" + basepath))
        sf.write(writepath, data, sr)
