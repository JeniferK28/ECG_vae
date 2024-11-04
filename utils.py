from __future__ import division, print_function
#from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, classification_report
import os
import h5py

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)


def add_noise(config):
    noises = dict()
    noises["trainset"] = list()
    noises["testset"] = list()
    import csv
    try:
        testlabel = list(csv.reader(open('C:/Users/Jenifer/Desktop/ecg/training2017/REFERENCE.csv')))
    except:
        cmd = "curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)
        os.system("unzip training2017.zip")
        testlabel = list(csv.reader(open('C:/Users/Jenifer/Desktop/ecg/training2017/REFERENCE.csv')))
    for i, label in enumerate(testlabel):
      if label[1] == '~':
        filename = 'C:/Users/Jenifer/Desktop/ecg/training2017/data/'+ label[0] + '.mat'
        from scipy.io import loadmat
        noise = loadmat(filename)
        noise = noise['val']
        _, size = noise.shape
        noise = noise.reshape(size,)
        noise = np.nan_to_num(noise) # removing NaNs and Infs
        from scipy.signal import resample
        noise= resample(noise, int(len(noise) * 360 / 300) ) # resample to match the data sampling rate 360(mit), 300(cinc)
        from sklearn import preprocessing
        noise = preprocessing.scale(noise)
        noise = noise/1000*6 # rough normalize, to be improved
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(noise, distance=150)
        choices = 10 # 256*10 from 9000
        picked_peaks = np.random.choice(peaks, choices, replace=False)
        for j, peak in enumerate(picked_peaks):
          if peak > config.input_size//2 and peak < len(noise) - config.input_size//2:
              start,end  = peak-config.input_size//2, peak+config.input_size//2
              if i > len(testlabel)/6:
                noises["trainset"].append(noise[start:end].tolist())
              else:
                noises["testset"].append(noise[start:end].tolist())
    return noises

def preprocess(data, config):
    sr = config.sample_rate
    if sr == None:
      sr = 300
    data = np.nan_to_num(data) # removing NaNs and Infs
    from scipy.signal import resample
    data = resample(data, int(len(data) * 360 / sr) ) # resample to match the data sampling rate 360(mit), 300(cinc)
    from sklearn import preprocessing
    data = preprocessing.scale(data)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(data, distance=150)
    data = data.reshape(1,len(data))
    data = np.expand_dims(data, axis=2) # required by Keras
    return data, peaks

# predict
def uploadedData(filename, csvbool = True):
    if csvbool:
      csvlist = list()
      with open(filename, 'r') as csvfile:
        for e in csvfile:
          if len(e.split()) == 1 :
            csvlist.append(float(e))
          else:
            csvlist.append(e)
    return csvlist

def read_data(path, label_path):
    data=[]
    f_s=[]
    label=[]
    f=300
    s=[]
    all_label=pandas.read_csv(label_path)
    for i in range (len(all_label)):
        txt=str(all_label.values[i]).split(';')
        name=txt[0].split('A')
        D = loadmat(path+name[1])
        d_t=D['val']

        # Preprocessing
        fs, s = signal.periodogram(d_t[0,0:2700],f)

        #s=d_t[0,0:2700]
        s=np.reshape(s,(1,1,np.shape(s)[0]))
        f_s.append(f)
        data.append(s)
        l = [char for char in txt[1]]
        if l[0]== 'N':
            label.append(0)
        elif  l[0]== 'A':
            label.append(1)
        else: label.append(2)
    return data, label, f_s

def read_data_ecg(path):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    return data

def computeAUROC(dataGT, dataPRED, classCount,totalval):
    outAUROC = []
    outAUPRC = []
    outAP = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    #n=datanpGT.shape
    data=np.zeros((totalval,6))
    for i in range(totalval):
        if datanpGT[i]==0: data[i,0]=1
        elif datanpGT[i] == 1:data[i, 1] = 1
        elif datanpGT[i]==2: data[i,2]=1
        elif datanpGT[i] == 3:data[i, 3] = 1
        elif datanpGT[i] == 4: data[i, 4] = 1
        elif datanpGT[i] == 5: data[i, 5] = 1


    for i in range(classCount):
        outAUROC.append(roc_auc_score(data[:, i], datanpPRED[:, i], average='weighted'))
        outP, outR, _ = prc(data[:, i], datanpPRED[:, i])
        outAUPRC.append(auc(outR, outP))
        outAP.append(ap(data[:, i], datanpPRED[:, i]))

    return outAUROC, outAUPRC, outAP


