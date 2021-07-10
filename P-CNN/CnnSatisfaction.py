#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:54:02 2020

@author: D. Escobar-Grisales
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn import metrics
from pathlib import Path
import seaborn as sns
from sklearn.model_selection import train_test_split
#%%
plt.style.use('ggplot')
def plot_history(history,parameters, name_folder):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    name = '_'.join(parameters)
    Path(name_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(name_folder+'/'+name+'.png')




class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 regularization_rate=1e-4,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim, mask_zero=True)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
#                                    kernel_regularizer=regularizers.l2(regularization_rate))
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
#                                     kernel_regularizer=regularizers.l2(regularization_rate))
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
#                                     kernel_regularizer=regularizers.l2(regularization_rate))
        self.pool = layers.GlobalMaxPool1D() 
        
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
#                                 kernel_regularizer=regularizers.l2(regularization_rate))
        self.dropout = layers.Dropout(rate=dropout_rate)

        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x_questions = self.embedding(inputs[0])
        x_questions1 = self.bigram(x_questions)
#        x_1 = self.dropout(x_1)
        x_questions1 = self.pool(x_questions1)
        x_questions2 = self.trigram(x_questions)
#        x_2 = self.dropout(x_2)
        x_questions2 = self.pool(x_questions2)
        x_questions3 = self.fourgram(x_questions)
#        x_3 = self.dropout(x_3)
        x_questions3 = self.pool(x_questions3)


        x_anwers = self.embedding(inputs[1])
        x_anwers1 = self.bigram(x_anwers)
#        x_1 = self.dropout(x_1)
        x_anwers1 = self.pool(x_anwers1)
        x_anwers2 = self.trigram(x_anwers)
#        x_2 = self.dropout(x_2)
        x_anwers2 = self.pool(x_anwers2)
        x_anwers3 = self.fourgram(x_anwers)
#        x_3 = self.dropout(x_3)
        x_anwers3 = self.pool(x_anwers3)
        
        merged = tf.concat([x_questions1, x_questions2, x_questions3,
                            x_anwers1, x_anwers2, x_anwers3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
#        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output
def metrics_loss(loss, threshold = 1e-3):
    loss_best = np.where(np.asarray(loss) <= threshold)[0]
    epochs_good_loss = len(loss_best)
    return epochs_good_loss

    
        

# Solo PAN17

pathBase = 'PathDataBasePreprocesed-csv--format'  
pathResults='path-Results--format csv'  
database = pd.read_csv(pathBase)
 


ids_conversation = np.asarray(database['ids'])
questions = np.asarray(database['Questions'])
answers = np.asarray(database['Answers'])
labels = np.asarray(database['Labels'])
#%%

for itr in range(0,1):

    idsTrain, idsTest, labelTrain, labelTest = train_test_split( ids_conversation, labels, test_size=0.3, shuffle = True)
    idxTrain = [np.where(ids_conversation == i)[0][0] for i in idsTrain]
    idxTest = [np.where(ids_conversation == i)[0][0] for i in idsTest]
    
    questionsTrain, questionsTest = questions[idxTrain], questions[idxTest]
    answersTrain, answersTest = answers[idxTrain], answers[idxTest]
    labelTrain, labelTest = labels[idxTrain], labels[idxTest]
    
    vocabularyComplete = np.hstack((questionsTrain, answersTrain))
    
    vocabulaty_size = 2**10
    
    # # Tokenizacion
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(vocabularyComplete, target_vocab_size= vocabulaty_size)
    data_questions_train = [tokenizer.encode(sentence) for sentence in questionsTrain]
    data_answers_train = [tokenizer.encode(sentence) for sentence in answersTrain]
    data_questions_test = [tokenizer.encode(sentence) for sentence in questionsTest]
    data_answers_test = [tokenizer.encode(sentence) for sentence in answersTest]
    #Padding
    #%%
    MAX_LEN_question = max([len(sentence) for sentence in data_questions_train])
    MAX_LEN_answers = max([len(sentence) for sentence in data_answers_train])
    #MAX_LEN = 630
    data_question_train = tf.keras.preprocessing.sequence.pad_sequences(data_questions_train,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN_question)
    data_answers_train = tf.keras.preprocessing.sequence.pad_sequences(data_answers_train,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN_answers)
    
    
    
    data_question_test = tf.keras.preprocessing.sequence.pad_sequences(data_questions_test,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN_question)
    data_answers_test = tf.keras.preprocessing.sequence.pad_sequences(data_answers_test,
                                                                value=0,
                                                                padding="post",
                                                                maxlen=MAX_LEN_answers)

    data_inputs_train = [data_question_train, data_answers_train]
    data_inputs_test = [data_question_test, data_answers_test]


    parameter_grid ={'EMB_DIM':[100],
                 'NB_FILTERS': [8, 16, 32, 64, 128, 256],
                 'units_dense': [8, 16, 32, 64, 128, 256],
                 'dropout': [0],
                 'learning_rate': [1e-4],
                 'regularization_rate': [0],
                 'batch_size': [32]}
    results = []
    accuracy_final = []
    accuracy_final_mean = []
    accuracy_final_mode = []
    combinaciones = []
    
    lossByEpoch1 = []
    lossByEpoch2 = []
    for dropout in parameter_grid['dropout']:
        for regularization_r in parameter_grid['regularization_rate']:
            for l_r in parameter_grid['learning_rate']:
                for emd_d in parameter_grid['EMB_DIM']:
                    for n_filters in parameter_grid['NB_FILTERS']:
                        for n_units in parameter_grid['units_dense']:
                    
                            VOCAB_SIZE = vocabulaty_size + 1 # 65540 # +1
                            
                            EMB_DIM = emd_d
                            NB_FILTERS = n_filters
                            FFN_UNITS = n_units
                            NB_CLASSES = 2#len(set(train_labels))
                            learning_r = l_r
                            DROPOUT_RATE = dropout
                            BATCH_SIZE = parameter_grid['batch_size'][0]
                            NB_EPOCHS = 70
                            REGULATION_RATE = regularization_r
                            
                            Dcnn = DCNN(vocab_size=VOCAB_SIZE,
                                        emb_dim=EMB_DIM,
                                        nb_filters=NB_FILTERS,
                                        FFN_units=FFN_UNITS,
                                        nb_classes=NB_CLASSES,
                                        dropout_rate=DROPOUT_RATE,
                                        regularization_rate= regularization_r)
                            
                            if NB_CLASSES == 2:
                                opt = tf.keras.optimizers.Adam(learning_rate=l_r)
                                Dcnn.compile(loss="binary_crossentropy",
                                             optimizer=opt,
                                             metrics=["accuracy"])
                            else:
                                opt = tf.keras.optimizers.Adam(learning_rate=l_r)
                                Dcnn.compile(loss="sparse_categorical_crossentropy",
                                             optimizer=opt,
                                             metrics=["sparse_categorical_accuracy"])
              
                            checkpoint = tf.keras.callbacks.ModelCheckpoint(pathResults+"/best_model.hdf5",
                                                                            monitor='val_loss', verbose=1,save_best_only=True, mode='auto', period=1)
                            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            history = Dcnn.fit(data_inputs_train,
                                     labelTrain,
                                     batch_size=BATCH_SIZE,
                                     epochs=NB_EPOCHS,
                                     validation_data=(data_inputs_test, labelTest),
                                     callbacks = [checkpoint])
                                      #callbacks = [checkpoint, early_stopping])
                    
                            plot_history(history, ['n_filters'+str(n_filters), 'n_units'+str(n_units)], 
                                          pathResults)
                            # plot_history(history, ['dropout:'+str(dropout), 'regularization_r'+str(regularization_r)], 
                            #               pathResults)
                                                      
                            lossByEpoch1.append(metrics_loss(history.history['loss'], threshold=0.01))
                            lossByEpoch2.append(metrics_loss(history.history['loss'], threshold=0.001))
                            test_labels = labelTest.astype(float)
                            Dcnn.load_weights(pathResults+"/best_model.hdf5")
                            scores = Dcnn.predict(data_inputs_test)
                            scores_pred = np.round(scores)
                            acc = metrics.accuracy_score(test_labels, scores_pred)
    # #                        predicciones = np.round(scores
    #                         # sensibilidad = metrics.precision_score(predicciones, test_labels)
    #                         # recall = metrics.recall_score(predicciones, test_labels)
    #                         # f1_score = metrics.f1_score(scores_pred, test_labels)
    #                         # fpr, tpr, thresholds = metrics.roc_curve(test_labels, scores)
    #                         # auc = metrics.auc(fpr,tpr)
                            combinaciones.append([n_filters, n_units])
    #                         #results.append([predicciones, sensibilidad, recall, f1_score])
                            accuracy_final.append(acc)
                            
                            
                            # databaseOriginal = pd.read_csv(pathBase + 'TestOrganizadaPreprosada.csv') 
                            # y_pred_mean, y_pred_mode, y_real = compute_metricsbysubject(scores, predicciones, ids_test_gender, databaseOriginal)
                            # acc_mean = metrics.accuracy_score(y_real, y_pred_mean)
                            # acc_mode = metrics.accuracy_score(y_real, y_pred_mode)
                            # accuracy_final_mean.append(acc_mean)
                            # accuracy_final_mode.append(acc_mode)

    
