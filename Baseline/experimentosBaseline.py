#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:41:06 2020

@author: D. Escobar-Grisales
"""

import cross_validation_with_greadSerach as svm_funtion
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sb
#%%

def balanceconversations (Features, Labels):
    
    indexQst = np.arange(0, Features.shape[0], 2)
    
    indexAns = np.arange(1, Features.shape[0], 2) 
    
    FeaturesQst = Features[indexQst]
    
    FeaturesAns = Features[indexAns]
    
    Labels = Labels[indexQst]
    
    max_conversations = min([len(np.where(Labels ==1)[0]), len(np.where(Labels ==0)[0])])
    indexSts = np.where(Labels ==1)[0][0:max_conversations]
    indexNonSts = np.where(Labels == 0)[0][0:max_conversations]
    
    indexF = np.hstack((indexSts, indexNonSts))
    
    FeaturesAns = FeaturesAns[indexF]
    FeaturesQst = FeaturesQst[indexF]
    
    Labels = Labels[indexF]
    
    
    FeaturesFinal = np.concatenate((FeaturesQst.T, FeaturesAns.T)).T

    return FeaturesFinal, Labels
#%%
def grafic_results(y_real, y_scores, y_pred):

    fpr, tpr, thresholds = roc_curve(y_real, y_scores)
    AUC = auc(fpr, tpr)
    #plt.figure(figsize = (10,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label = 'AUC: '+ '%.2f' % AUC  )
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    cm_normalize = []
    c_m=confusion_matrix(y_real,y_pred)
    cm_normalize.append(c_m[0]/(len(y_real)/2))
    cm_normalize.append(c_m[1]/(len(y_real)/2))
    cm_normalize = np.vstack(cm_normalize)
    plt.figure()
    heat_map = sb.heatmap(cm_normalize, annot=True)

#%%
print('-'*20)

FeaturesChatbot = np.asarray(np.loadtxt('path-FeaturesMatrix-chatbot1'))
LabelsProteccion = np.asarray(np.loadtxt('path-Label-chatbot1'))

Features_BalanceProteccion, Labels_BalanceProteccion = balanceconversations(FeaturesProteccion, LabelsProteccion)

svm_funtion.Cross_validation_SVM(Features_BalanceProteccion, Labels_BalanceProteccion, 5,'results.csv', 5)
print('-'*20)






