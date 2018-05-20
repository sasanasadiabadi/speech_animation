import numpy as np
import numpy.matlib
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################### set parameters #############################
path = '../path/to/data'

Kx = 11
Ky = 5
num_pca = 16
mode = 1 # 1 for CNN / 0 for MLP

######################### read data from file ##########################

def read_data(path):
  Xs = []
  ys = []

  files = os.listdir(path)
  for name in files:
      data = pd.read_csv(os.path.join(path,name),header=None)
      XX = data.iloc[:41,:].values
      yy = data.iloc[41:,:].values

      Xs.extend(XX.T)
      ys.extend(yy.T)
  
  return np.transpose(Xs) np.transpose(y)

X, y = read_data(path)

######################### create dataset for training ####################

def create_dataset(Xs,ys,Kx,Ky,mode):
  kx = int((Kx-1)/2)
  ky = int((Ky-1)/2)

  # padd data to begining and end
  XX = np.hstack((np.repeat(Xs[:,[0]],kx,axis=1), Xs, np.repeat(Xs[:,[-1]],kx,axis=1)))
  yy = np.hstack((np.repeat(ys[:,[0]],ky,axis=1), ys, np.repeat(ys[:,[-1]],ky,axis=1)))

  Xphn, ypca = ([] for i in range(2))
  
  for i in range(kx,np.shape(XX)[1]-kx):
      if mode:
          tmp = XX[:,i-kx:i+kx+1]
          tmp = tmp[:,np.newaxis,:]
          Xphn.append(tmp)
      else:
          tmp = np.reshape(XX[:,i-kx:i+kx+1],(1,41*Kx),order='F')
          Xphn.extend(tmp)

  for j in range(ky,np.shape(yy)[1]-ky):
     tmp = np.reshape(yy[:,j-ky:j+ky+1],(1,num_pca*Ky),order='F')
     ypca.extend(tmp)
  
  return np.array(Xphn), np.array(ypca) 

Xphn, ypca = create_dataset(X,y,Kx,Ky,mode)

print(np.shape(Xphn))
print(np.shape(ypca))

def train_model(mode):
    model = Sequential()
    
    if mode:
      model.add(Conv2D(256,kernel_size=(7,1),padding='same',activation='tanh',input_shape=(41,1,Kx)))
      model.add(MaxPool2D(pool_size=(4,1),strides=(2,1)))
      #model.add(Dropout(0.25))

      model.add(Conv2D(512,kernel_size=(5,1),padding='same',activation='tanh'))
      model.add(MaxPool2D(pool_size=(2,1),strides=(2,1)))
      #model.add(Dropout(0.25))

      #model.add(Conv2D(512,kernel_size=(3,1),padding='valid',activation='tanh'))
      #model.add(MaxPool2D(pool_size=(2,1),strides=(2,1)))
      #model.add(Dropout(0.25))

      model.add(Flatten())
      model.add(Dense(3000,activation='tanh'))
      model.add(Dropout(0.5))

      model.add(Dense(3000,activation='tanh'))
      model.add(Dropout(0.5))

      model.add(Dense(16*Ky))
      model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    else:
      model = Sequential()
      model.add(Dense(3000,activation='tanh',input_dim=41*Kx))
      model.add(Dropout(0.5))
      model.add(Dense(3000,activation='tanh'))
      model.add(Dropout(0.5))
      model.add(Dense(3000, activation='tanh'))
      model.add(Dropout(0.5))

      model.add(Dense(16*Ky))
      model.compile(loss='mse',optimizer='adam',metrics=['mse'])

    return model

######################## model train #######################################

Xtrn, Xtst, ytrn, ytst = train_test_split(Xphn,ypca,test_size=0.1,shuffle=True,random_state=42)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto')
mdl_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')

model = train_model(mode)
model.fit(Xtrn,ytrn,validation_data=(Xtst,ytst),batch_size=128,epochs=10,shuffle=True,verbose=2,
          callbacks=[early_stop, mdl_save])
          
### save model
model_json = model1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Model saved to disk")



