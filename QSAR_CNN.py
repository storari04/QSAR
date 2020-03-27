from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.cross_validation import train_test_split
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

rows = df_bind.shape[ 0 ]
mols = [ ]
act = [ ]
X = [ ]

def generate_fparr( mol ):
    arr = np.zeros( (1,) )
    fp = AllChem.GetMorganFingerprintAsBitVect( mol, 2, nBits = 1024, useFeatures = True )
    DataStructs.ConvertToNumpyArray( fp, arr )
    size = 32
    return arr.reshape( 1, size, size )

def save_fig( fparr, filepath, size=32 ):
    X, Y = np.meshgrid( range(size), range(size) )
    Z = fparr
    Z = Z[::-1,:]
    plt.xlim( 0, 31 )
    plt.ylim( 0, 31 )
    plt.pcolor(X, Y, Z[0])
    plt.gray()
    plt.savefig( filepath )
    plt.close()

def act2bin( val ):
    if val > 10000:
        return 0
    else:
        return 1

for i in range( rows ):
    try:
        smi = df_bind.CANONICAL_SMILES[i]
        mol = Chem.MolFromSmiles( smi )
        if mol != None:
            mols.append( mol )
            act.append( act2bin( df_bind.STANDARD_VALUE[i]) )
        else:
            pass
    except:
        pass

# save mols image dataset
for idx, mol in enumerate( mols ):
    X.append( generate_fparr( mol ) )
    if act[ idx ] == 1:
        save_fig( X[ idx ], "./posi/idx_{}.png".format( idx ) )
    elif act[ idx ] == 0:
        save_fig( X[ idx ], "./nega/idx_{}.png".format( idx ) )


X = np.asarray(X)
Y = np.asarray(act)

x_train, x_test, y_train, y_test = train_test_split( X,Y, test_size=0.2, random_state=123 )

f = open( 'fpimagedataset.pkl', 'wb' )
pickle.dump([ ( x_train,y_train ), ( x_test, y_test ) ], f)
f.close()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt

( x_train, y_train ), ( x_test, y_test ) = pickle.load( open('fpimagedataset.pkl', 'rb') )

batch_size = 200
nb_classes = 2
nb_epoch = 100
nb_filters = 32
nb_pool = 2
nb_conv = 3
im_rows , im_cols = 32, 32
im_channels = 1

train_x = x_train.astype('float32')
train_x = x_test.astype('float32')
train_y = np_utils.to_categorical( y_train, nb_classes )
test_y = np_utils.to_categorical( y_test, nb_classes )

print( x_train.shape[0], 'train samples' )
print( x_test.shape[0], 'test samples' )

model = Sequential()
model.add(Convolution2D(32,3,3, input_shape = ( im_channels, im_rows, im_cols )))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, ))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D( pool_size=( 2, 2 ) ) )
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile( loss='categorical_crossentropy',
               optimizer='adadelta',
               metrics=['accuracy'],
               )
history = model.fit(train_x, train_y, batch_size = batch_size,
                  nb_epoch = nb_epoch,
                  verbose = 1,
                  validation_data = (test_x, test_y))
               )
print(model.summary())
score = model.evaluate(test_x, test_y, verbose = 0)

loss = history.history['loss']
acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
plt.plot( range(len( loss )), loss, label='loss' )
plt.plot( range(len( val_loss )), val_loss, label='val_loss' )
plt.xlabel( 'epoch' )
plt.ylabel( 'loss' )
plt.savefig( 'loss.png' )
plt.close()
plt.plot( range(len( acc )), acc, label='accuracy' )
plt.plot( range(len( val_acc )), val_acc, label='val_accuracy' )
plt.xlabel( 'epoch' )
plt.ylabel( 'acc' )
plt.savefig( 'acc.png' )
plt.close()
