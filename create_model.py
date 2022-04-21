import librosa, numpy as np, csv, os
from concurrent import futures
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
from sklearn.svm import SVC

TONES = 12
CHORDS = 13

filename = "./datalist.csv"
base = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.normpath(os.path.join(base, filename))

chord_dic = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
  12: "Cm", 13: "C#m", 14: "Dm", 15: "D#m", 16: "Em", 17: "Fm", 18: "F#m", 19: "Gm", 20: "G#m", 21: "Am", 22: "A#m", 23: "Bm",
  24: "Cdim", 25: "C#dim", 26: "Ddim", 27: "D#dim", 28: "Edim", 29: "Fdim", 30: "F#dim", 31: "Gdim", 32: "G#dim", 33: "Adim", 34: "A#dim", 35: "Bdim",
  36: "Caug", 37: "C#aug", 38: "Daug", 39: "D#aug", 40: "Eaug", 41: "Faug", 42: "F#aug", 43: "Gaug", 44: "G#aug", 45: "Aaug", 46: "A#aug", 47: "Baug",
  48: "Csus4", 49: "C#sus4", 50: "Dsus4", 51: "D#sus4", 52: "Esus4", 53: "Fsus4", 54: "F#sus4", 55: "Gsus4", 56: "G#sus4", 57: "Asus4", 58: "A#sus4", 59: "Bsus4",
  60: "Csus2", 61: "C#sus2", 62: "Dsus2", 63: "D#sus2", 64: "Esus2", 65: "Fsus2", 66: "F#sus2", 67: "Gsus2", 68: "G#sus2", 69: "Asus2", 70: "A#sus2", 71: "Bsus2",
  72: "C7", 73: "C#7", 74: "D7", 75: "D#7", 76: "E7", 77: "F7", 78: "F#7", 79: "G7", 80: "G#7", 81: "A7", 82: "A#7", 83: "B7",
  84: "Cm7", 85: "C#m7", 86: "Dm7", 87: "D#m7", 88: "Em7", 89: "Fm7", 90: "F#m7", 91: "Gm7", 92: "G#m7", 93: "Am7", 94: "A#m7", 95: "Bm7",
  96: "C7sus4", 97: "C#7sus4", 98: "D7sus4", 99: "D#7sus4", 100: "E7sus4", 101: "F7sus4", 102: "F#7sus4", 103: "G7sus4", 104: "G#7sus4", 105: "A7sus4", 106: "A#7sus4", 107: "B7sus4",
  108: "C7sus2", 109: "C#7sus2", 110: "D7sus2", 111: "D#7sus2", 112: "E7sus2", 113: "F7sus2", 114: "F#7sus2", 115: "G7sus2", 116: "G#7sus2", 117: "A7sus2", 118: "A#7sus2", 119: "B7sus2",
  120: "Cdim7", 121: "C#dim7", 122: "Ddim7", 123: "D#dim7", 124: "Edim7", 125: "Fdim7", 126: "F#dim7", 127: "Gdim7", 128: "G#dim7", 129: "Adim7", 130: "A#dim7", 131: "Bdim7",
  132: "Caug7", 133: "C#aug7", 134: "Daug7", 135: "D#aug7", 136: "Eaug7", 137: "Faug7", 138: "F#aug7", 139: "Gaug7", 140: "G#aug7", 141: "Aaug7", 142: "A#aug7", 143: "Baug7",
  144: "C5", 145: "C#5", 146: "D5", 147: "D#5", 148: "E5", 149: "F5", 150: "F#5", 151: "G5", 152: "G#5", 153: "A5", 154: "A#5", 155: "B5",
}

chord_label = dict(zip(chord_dic.values(), chord_dic.keys()))

namechange = {
        "major": "", "minor": "m","dim": "dim", "aug": "aug", "sus4": "sus4",
        "sus2": "sus2", "major_7": "7", "minor_7": "m7", "dim_7": "dim7",
        "aug_7": "aug7", "sus4_7": "7sus4", "sus2_7": "7sus2", "power": "5",
      }

with open(filepath, "r") as f:
  reader = csv.reader(f)
  list = [row for row in reader]
  inst_l = list[0]
  tone_l = list[1]
  chord_l = list[2]
  group = list[3]

  chord_array = np.zeros((TONES, CHORDS, TONES))
  label_array = [[],[],[],[],[],[],[],[],[],[],[],[]]

  def analysis(index : int):
    chord_n = 0
    tone = tone_l[index]
    
    labels = []
    for chord in chord_l:
      for inst in inst_l:
        basepath = tone + "/" + inst + "/" + chord + ".wav"
        path = os.path.normpath(os.path.join(base, group[0] + "/" + basepath))

        data, sr = librosa.load(path)
        raw_chroma = librosa.feature.chroma_cqt(y = data, sr = sr)

        for k in range(0, len(raw_chroma[0])):
          chroma = np.array(raw_chroma[:, k])
          chord_array[index, chord_n, :] += chroma.T

      labels.append(str(tone) + str(namechange[chord]))
      chord_n += 1

    label_array[index] = labels
  
  future_list = []
  with futures.ThreadPoolExecutor(max_workers=12) as executor:
    for i in range(12):
      future = executor.submit(analysis, index = i)
      future_list.append(future)
      _ = futures.as_completed(fs=future_list)
  
  print("chord analysis is completed")

  chord_data = []
  label = []
  for label_a in label_array:
    for l in label_a:
      label.append(l)
  
  for chords in chord_array:
    for chord in chords:
      chord_np = np.array(chord)
      chord_np /= np.linalg.norm(chord_np, ord=2)
      chord_data.append(chord_np)
  
  label_id = []
  for l in label:
    label_id.append(chord_label[l])

  print(label_id)

  model = LogisticRegression()
  model.fit(chord_data, label_id)

  filename = "chord_model.sav"
  pickle.dump(model, open(filename, 'wb'))

  print("model create success")
