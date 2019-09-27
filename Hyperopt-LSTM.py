#!/usr/bin/env python
# coding: utf-8

# ### Motivation and assumptions

# Using the following procedure, you can easily build and test LSTM Networks.<br>
# Hyperparameters are set using Hyperopt Python module.
# 
# There are some assumptions to remember regarding this particular example:
# 
# - Data are contained in .csv files. Please note, that if you created the .csv with Excel, it can use ";" instead of "," as value delimiter. In this case replace all ";" with ",".
# - A complete usage of these functions expects to create two folders where any number of .csv files can be placed. The first folder is used to train AND validate the algorithm. The second folder is used to test the accuracy of our model.
# - The first row of all .csv files must be the column names. Each column corresponds to an input variable. Each row (from the 2nd on) is a set of such input variables values.
# - It follows that all .csv files must have the same number of columns, but can have a different row number.
# - By putting different .csv files in the same folder, you let the program merge them as they would be a "single" bigger .csv file.

# ### Libraries

# In[15]:


import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from hyperopt import fmin, hp, tpe, STATUS_OK, space_eval, Trials
from keras import backend, optimizers
import pickle


# ### Define your architecture

# In this example we are going to optimize a feed forward neural network. It's a <b>regression</b> problem.

# #### Settings

# In[16]:


training_folder = 'train' # we store the .csv training data
# in the "train" folder. All .csv will be merged together into a single
# dataset


# #### Hyperparameters

# In[17]:


hyper_space = {
    'lstm_units_1': hp.choice('lstm_units_1', np.arange(0, 30, 1)),
    'lstm_units_2': hp.choice('lstm_units_2', np.arange(0, 30, 1)),
    'lstm_units_3': hp.choice('lstm_units_3', np.arange(0, 30, 1)),
    'lstm_units_4': hp.choice('lstm_units_4', np.arange(0, 30, 1)),
    'dense_units_1': hp.choice('dense_units_1', np.arange(0, 8, 1)),
    'dense_units_2': hp.choice('dense_units_2', np.arange(0, 8, 1)),
    'dense_units_3': hp.choice('dense_units_3', np.arange(0, 8, 1)),
    'dense_units_4': hp.choice('dense_units_4', np.arange(0, 8, 1)),
    'dense_units_5': hp.choice('dense_units_5', np.arange(0, 8, 1)),
    'lstm_dropout_1' : hp.uniform('lstm_dropout_1', 0, 0.9),
    'lstm_dropout_2' : hp.uniform('lstm_dropout_2', 0, 0.9),   
    'lstm_dropout_3' : hp.uniform('lstm_dropout_3', 0, 0.9),
    'lstm_dropout_4' : hp.uniform('lstm_dropout_4', 0, 0.9),
    'dense_dropout_1' : hp.uniform('dense_dropout_1', 0, 0.9),
    'dense_dropout_2' : hp.uniform('dense_dropout_2', 0, 0.9),   
    'dense_dropout_3' : hp.uniform('dense_dropout_3', 0, 0.9),
    'dense_dropout_4' : hp.uniform('dense_dropout_4', 0, 0.9),
    'rec_dropout_1': hp.uniform('rec_dropout_1', 0, 0.9),
    'rec_dropout_2': hp.uniform('rec_dropout_2', 0, 0.9),
    'rec_dropout_3': hp.uniform('rec_dropout_3', 0, 0.9),
    'rec_dropout_4': hp.uniform('rec_dropout_4', 0, 0.9),
    'batch_size': hp.choice('batch_size', np.arange(1,66,16)),
    'timesteps': hp.choice('timesteps', np.arange(1,10,1))
}


# #### Data preparation

# In[18]:


def serie_shift(dataset, column_name, timesteps):
    data = pd.DataFrame()
    data[column_name] = dataset[column_name]
    for i in range(1, timesteps):
        data['%s+%d' % (column_name, i)] = data[column_name].shift(-1*i)
    data = data.dropna()
    return data

def convert_date(df, df_col, date_format = '%Y-%m-%d'):
    df[df_col] = df[df_col].apply(
        lambda el: int(datetime.strptime(el,date_format).timestamp()/86400))
    df[df_col] -= df[df_col].min() # shifting (to start from 0)
    return df

def get_data(folder):
    df_list = []
    for f in glob.glob(os.path.join(folder,'*.csv')):
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list)
    df = df.dropna() # we filter out rows with non valid values
    df = convert_date(df,"Date") 
    df = df.astype(np.float64)
    return df

def normalize_data(df):
    for columnName in df:
        df[columnName] -= df[columnName].mean()
        df[columnName] /= df[columnName].std()
    return df

## OPTIONAL, not used here
def get_random_data(cols, samples, timesteps): # cols the columns lists
    randata = np.random.random((samples, timesteps))
    randata = pd.DataFrame(data=randata, index=np.arange(0,1000), columns=cols)
    return randata

# LSTM for BINARY CLASSIFICATION problem
def lstm_in_out(df, output_column_number, timesteps):
    # output_column_number starts from 0 (-> first column)
    tmp = []
    for columnName in df:
        tmp.append(serie_shift(df, columnName, timesteps+1).to_numpy())        
    tmp = tuple(tmp)
    X = np.dstack(tmp)
    Y = X[:,-1,1:2] # timestep+1 assigned to Y
    X = np.delete(X, -1, axis=1) # timestep+1 deleted from X
    diff = Y-X[:,-1,output_column_number:output_column_number+1]
    Y = np.where(diff >= 0, [1,0], [0,1])
    # [1,0] => next number is higher or equal
    # [0,1] => next number is lower
    return X,Y


# In[19]:


df = get_data(training_folder)


# #### Model Function

# In[20]:


def train_hyper_model(dataFrame, hyper_params):
    print(hyper_params)
    X,Y = lstm_in_out(dataFrame, 1,hyper_params['timesteps'])
    print(X.shape[2])
    model = Sequential()
   
    model.add(LSTM(
        units=hyper_params['lstm_units_1'],
        activation='selu',
        recurrent_dropout=hyper_params['rec_dropout_1'],
        return_sequences = True,
        input_shape=(hyper_params['timesteps'], X.shape[2])
    ))

    model.add(Dropout(hyper_params['lstm_dropout_1']))
    
    model.add(LSTM(
        units=hyper_params['lstm_units_2'],
        recurrent_dropout=hyper_params['rec_dropout_2'],
        return_sequences = True,
        activation='selu',
    ))

    model.add(Dropout(hyper_params['lstm_dropout_2']))

    
    model.add(LSTM(
        units=hyper_params['lstm_units_3'],
        recurrent_dropout=hyper_params['rec_dropout_3'],
        return_sequences = True,
        activation='selu',
    ))
    
    model.add(Dropout(hyper_params['lstm_dropout_3']))

    
    model.add(LSTM(
        units=hyper_params['lstm_units_4'],
        recurrent_dropout=hyper_params['rec_dropout_4'],
        return_sequences = False,
        activation='selu',
    ))
    
    model.add(Dropout(hyper_params['lstm_dropout_4']))

    model.add(Dense(units=hyper_params['dense_units_1'], activation='selu'))

    model.add(Dropout(hyper_params['dense_dropout_1']))
    
    model.add(Dense(units=hyper_params['dense_units_2'], activation='selu'))
    
    model.add(Dropout(hyper_params['dense_dropout_2']))
    
    model.add(Dense(units=hyper_params['dense_units_4'], activation='selu'))

    model.add(Dropout(hyper_params['dense_dropout_3']))
    
    model.add(Dense(units=hyper_params['dense_units_5'], activation='selu'))
    
    model.add(Dense(units=2,activation="softmax"))
    
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    history = model.fit(
        X,
        Y,
        batch_size=hyper_params['batch_size'],
        validation_split=0.2,
        epochs = 20,
        shuffle = True,
        verbose=0)
    
    # take the last 8 accuracy values, and return their mean value:
    return np.mean(history.history['val_acc'][-8:])


# #### Objective function

# In[21]:


def hyperopt_fn(hyper_params):
    accuracy = train_hyper_model(df, hyper_params) # X,Y are globally defined!
    backend.clear_session() # clear session to avoid models accumulation in memory
    return {'loss': -accuracy, 'status': STATUS_OK}


# <b>Note</b>: The STATUS_OK value is very important to avoid numerical errors problems produced by some particular set of (hyper)parameters values.

# ### Let's optimize!

# Thanks to Trials, which stores and track the progress, you have the possibility to execute a new optimization process, but starting from previous ones.

# In[22]:


keep_trials = Trials()

# we can also load trials from file using prickle:
f = open('store_trials_LSTM.pckl', 'rb')
keep_trials = pickle.load(f)
f.close()


# By setting the option trials = keep_trials, if you run again the same cell it will not compute any furter iteration, since it consider the previous ones as completed.
# For example, if you have done 10 iterations, than you change the iterations to 30 (max_evals = 30) and run the cell again, the optimization will perform 20 iteration (from 11 to 20!).
# If you want to reset the iteration after each code execution, just move the trials parameter.

# In[23]:


get_ipython().run_cell_magic('time', '', "while True:\n    try:\n        opt_params = fmin(\n                        fn=hyperopt_fn,\n                        space=hyper_space,\n                        algo=tpe.suggest,\n                        max_evals=18, # stop searching after 18 iterations\n                        trials = keep_trials\n                        )\n\n        # store trials in a file\n        f = open('store_trials_LSTM.pckl', 'wb')\n        pickle.dump(keep_trials, f)\n        f.close()\n\n        print(space_eval(hyper_space, opt_params))\n        print('number of trials:', len(keep_trials.trials))\n        break\n    except:\n        continue")


# In[24]:


print(keep_trials.trials[-1]['misc']['vals']) # print last hyperparameters value

