import matplotlib.pyplot as plt
import numpy as np
import librosa, os, csv, librosa.display

filename = "./datalist.csv"
base = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.normpath(os.path.join(base, filename))

with open(filepath, "r") as f:
  reader = csv.reader(f)
  list = [row for row in reader]
  inst_l = list[0]
  tone_l = list[1]
  chord_l = list[2]
  group = list[3]

  basepath = tone_l[0] + "/" + inst_l[0] + "/" + chord_l[0] + ".wav"
  print(basepath)
  data, sr = librosa.load(os.path.normpath(os.path.join(base, group[0] + "/" + basepath)))
  raw_chroma = librosa.feature.chroma_cqt(y = data, sr = sr)
  print(raw_chroma.shape)
  chord_data = np.zeros((raw_chroma.shape[0]))
  for i in range(0, raw_chroma.shape[0]):
    chord_data += raw_chroma[:, i]
  
  chord_data = np.array([chord_data])
  chord_data /= np.linalg.norm(chord_data, ord=2)
  print(chord_data.T.shape)
  plt.imshow(chord_data.T, cmap=plt.cm.jet, interpolation='nearest')
  plt.show()