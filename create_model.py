from logging import log
import sklearn, librosa, numpy as np, csv, os
from concurrent import futures

TONES = 12
CHORDS = 13

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

  chord_array = np.zeros((TONES, TONES, CHORDS))

  def analysis(index : int):
    chord_n = 0
    tone = tone_l[index]
    for chord in chord_l:
      for inst in inst_l:
        basepath = tone + "/" + inst + "/" + chord + ".wav"
        path = os.path.normpath(os.path.join(base, group[0] + "/" + basepath))

        data, sr = librosa.load(path)
        raw_chroma = librosa.feature.chroma_cqt(y = data, sr = sr)

        for k in range(0, len(raw_chroma[0])):
          chord_array[index, :, chord_n] += raw_chroma[:, k]
      
      chord_n += 1
  
  future_list = []
  with futures.ThreadPoolExecutor(max_workers=12) as executor:
    for i in range(12):
      future = executor.submit(analysis, index = i)
      future_list.append(future)
      _ = futures.as_completed(fs=future_list)
  
  print("chord analysis is completed")
  print(chord_array[0])
  
  