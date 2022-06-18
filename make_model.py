from pickletools import optimize
import numpy as np
import pandas as pd
from numpy import newaxis
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras import metrics
import time
import logging
import optuna
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix, recall_score, precision_score, f1_score

def data_split(data_np, seq_len, y):
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data_np) - sequence_length):
        result = np.append(result, data_np[index: index + sequence_length, :])
    result = np.reshape(result, (len(data_np) - sequence_length,sequence_length, data_np.shape[1]))
    row = round(0.9 * result.shape[0])
    train = result[:int(row),:,:]
    np.random.shuffle(train)
    x_train = train[:,:-1,:]
    y_train = train[:,-1,y]
    x_test = result[int(row):,:-1, :]
    y_test = result[int(row):, -1, y]

    return [x_train, y_train, x_test, y_test ]

def data_split_many_class(data_np, seq_len, y_num):
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data_np) - sequence_length):
        result = np.append(result, data_np[index: index + sequence_length, :])
    result = np.reshape(result, (len(data_np) - sequence_length,sequence_length, data_np.shape[1]))
    row = round(0.9 * result.shape[0])
    train = result[:int(row),:,:]
    np.random.shuffle(train)
    x_train = train[:,:-1,:]
    y_train = train[:,-1,-1*y_num:]
    x_test = result[int(row):,:-1, :]
    y_test = result[int(row):, -1, -1*y_num:]

    return [x_train, y_train, x_test, y_test ]

def build_model(layers):
    model = Sequential()

    model.add(LSTM(layers[2],input_shape = (layers[1], layers[0]),  return_sequences=True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(layers[3]))
    model.add(Activation("linear"))

    
    optimizer = "adam"
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'acc'])
    return model

def build_model_Binary_class(layers):
    model = Sequential()

    model.add(LSTM(layers[2],input_shape = (layers[1], layers[0]),  return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(layers[3]))
    model.add(Activation("sigmoid"))

    
    optimizer = "adam"
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['acc'])
    return model

def build_model_many_class(layers):
    model = Sequential()

    model.add(LSTM(layers[2],input_shape = (layers[1], layers[0]),  return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(layers[3]))
    model.add(Activation("softmax"))

    
    optimizer = "adam"
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
    return model

class Objective:
    def __init__(self, x_test, y_test,model):
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
    def __call__(self, trial):
        #optimizer_name = trial.suggest_categorical("optimizer", ["adam", "SGD", "RMSprop", "Adadelta"])
        epochs = trial.suggest_int("epochs", 5, 15,step=5, log=False)
        #batchsize = trial.suggest_int("batchsize", 8, 40,step=16, log=False)
        #activation = trial.suggest_categorical('activation', ['tanh'])
        
        
        
        self.model.fit(epochs)
        scores = cross_validate(self.model,
                                X=self.x_test, 
                                y=self.y_test,
                                scoring='accuracy')
        '''
        model.fit(x_train,y_train)
        val_acc = model.evaluate(x_test,y_test)[1]
        weights = model.get_weights()

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr(key="best_model_weights", value=weights)'''
        return scores

    def callback(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(key="best_model_weights", 
                                value=trial.user_attrs["best_model_weights"])
#objective = mm.Objective(x_test, y_test,model)
#study = optuna.create_study(direction='maximize') # 最大化
#study.optimize(objective,n_trials=100, timeout=None)

def draw_tran_result(model, x_test, y_test):
    out = model.predict(x_test)
    plt.figure()
    plt.plot(range(0, len(out)), out, color="b", label="actual")
    plt.plot(range(0, len(y_test)), y_test, color="r", label="expected")
    plt.legend()

def evalution(x_test, y_test,model,p=0.5):
    pred_y=[]
    for i in model.predict(x_test):
        if i>=p:pred_y=np.append(pred_y,1)
        else:pred_y=np.append(pred_y,0)
    confusion = confusion_matrix(y_test,pred_y)
    accuracy = (confusion[0][0]+confusion[1][1])/len(pred_y)
    precision = confusion[1][1]/(confusion[0][1]+confusion[1][1])
    recall = confusion[1][1]/(confusion[1][0]+confusion[1][1])
    F_value = 2*precision*recall/(precision+recall)
    print("混同行列{}".format(confusion))
    print("正解率は{} 適合率は{} 再現率は{} F値は{}".format(accuracy,precision,recall,F_value))

def evalution_many_class(x_test, y_test,model):
    pred_y=[]
    for i in range(len(y_test)):
        a=np.argmax(model.predict(x_test)[i])
        if a==0:
            pred_y=np.append(pred_y,[1,0,0])
        elif a==1:
            pred_y=np.append(pred_y,[0,1,0])
        else:
            pred_y=np.append(pred_y,[0,0,1])
    pred_y = pred_y.reshape(int(len(pred_y)/3),3)
    accuracy = accuracy_score(y_test, pred_y)
    confusion = confusion_matrix(y_test.argmax(axis=1), pred_y.argmax(axis=1))
    precision = precision_score(y_test, pred_y, average=None)
    recall = recall_score(y_test, pred_y, average=None)
    F_value = f1_score(y_test, pred_y, average=None)
    print("正解率は{}".format(accuracy))
    print("混同行列{}".format(confusion))
    print("正解率は{} \n適合率は{} \n再現率は{} \nF値は{}".format(accuracy,precision,recall,F_value))


def baibai(x_test, model,x_test_close,p=0.5):
    y_pred=[]
    for i in model.predict(x_test):
        if i>=p:y_pred=np.append(y_pred,1)
        else:y_pred=np.append(y_pred,0)
    position=0
    position_value=0
    value=0
    for i in range(len(y_pred)):
        if y_pred[i]==1:
            position_value+=x_test_close[i]
            position+=1
        elif y_pred[i]==0:
            if position==1: value=value+(x_test_close[i]*position-position_value)
            position=0
            position_value=0
    return value

def baibai_many_class(x_test, model,x_test_close):

    position=0
    position_value=0
    value=0
    for i in range(len(x_test_close)):
        if np.argmax(model.predict(x_test)[i])==2:
            position_value+=x_test_close[i]
            position+=1
        elif np.argmax(model.predict(x_test)[i])==1:
            if position==1: value=value+(x_test_close[i]*position-position_value)
            position=0
            position_value=0
        else:
            if position==1: value=value+(x_test_close[i]*position-position_value)
            position=0
            position_value=0
    return value

