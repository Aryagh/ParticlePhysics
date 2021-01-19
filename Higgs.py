# Particle Phyics Course Project by aryagh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

#read data
train_file = '../training.csv'
test_file = '../training.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

print('features: ',train.columns)
print('Labels: ',set(train['Label']))

#preprosses train data
le = preprocessing.LabelEncoder()
train['Label'] = le.fit_transform(train['Label'])

train.drop('Weight', axis=1,inplace=True)
X = train.loc[:,train.columns != 'Label']
Y = train.loc[:,'Label']

#preprosses test data
le = preprocessing.LabelEncoder()
test['Label'] = le.fit_transform(test['Label'])

test.drop('Weight', axis=1,inplace=True)
X_test = test.loc[:,train.columns != 'Label']
Y_test = test.loc[:,'Label']

#Visualization
import seaborn as sns
fig, ax = plt.subplots(5,6, figsize=(20, 15))
ax = ax.flatten()
for i in range(30):
    sns.distplot(X.iloc[:,i].values, ax=ax[i])
    ax[i].set_title(X.columns[i])
fig.tight_layout(pad=2.0)
#fig.show()

#DNN
from keras.models import Sequential
from keras.layers import Dense
#get number of columns in training data
n_cols = X.shape[1]

#add model layers
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

from keras.callbacks import EarlyStopping #set early stopping monitor so the model stops training when it won't improve anymore
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["acc"])
early_stopping_monitor = EarlyStopping(patience=3)#train model
model.fit(X, Y, validation_split=0.2, epochs=3, callbacks=[early_stopping_monitor],verbose=1)

#XGB
from xgboost import XGBClassifier

clf = XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=300,
                    n_jobs=-1, random_state=0, reg_alpha=0.2, 
                    colsample_bylevel=0.9, colsample_bytree=0.9)

print('XGB...')
clf.fit(X, Y)
print('XGB Accuracy:',accuracy_score(clf.predict(X_test), Y_test))
#save XGB model
# pickle.dump(clf, open("HIGGS.dat", "wb"))
