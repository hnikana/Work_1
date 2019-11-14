#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:54:59 2019

@author: hamedniakan
"""

import random
random.seed(30)
import ipywidgets as widgets
from IPython import display
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn import preprocessing
import category_encoders as ce
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf 
from sklearn.utils.class_weight import compute_class_weight 
import time
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn import preprocessing


from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression




from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.constraints import maxnorm # 
from sklearn.metrics import confusion_matrix

##################################################### control params  ######################################################################
imputation = False 
hashing = False 
onehot = True 

figfont_size = 18 
titlefont_size = 18 
labelfont_size = 16 
ticksize = 14

##################################################### reading data  ######################################################################

bank_additional_full = pd.read_csv('./bank-additional/bank-additional-full.csv' , sep =';') # reading datasets 
bank_additional = pd.read_csv('./bank-additional/bank-additional.csv', sep =';')

bank_additional.drop(['duration'] , axis = 1, inplace = True )                  # Dropping duration 
bank_additional_full.drop(['duration'] , axis = 1, inplace = True )


train_label = np.where(bank_additional_full['y'] == 'yes' , 1 , 0)   # creating labels
test_label = np.where(bank_additional['y'] == 'yes' , 1 , 0)

prediction = pd.DataFrame(test_label , columns = ['Actual'])  # collecting the actula & predictions from differnt models 

columns=list(bank_additional.columns)

#collectors
Recall_Precision_Fmeasure_Auc = {}
confusion_matrices = {}


##################################################### Visualization of the dataset  ######################################################################

# Visualization of the dataset 
plt.figure(1 , figsize = (15,10))
plt.suptitle('distribution of features ' , fontweight = 'bold' , fontsize = figfont_size , y = .95)
for i in range(len(bank_additional_full.columns)-1):
    ax = plt.subplot(4,5,i+1)
    ax.set_title(bank_additional_full.columns[i])
    for column in columns[:-1]:
        if bank_additional_full[column].dtypes != 'O' :
            	plt.hist([bank_additional_full[bank_additional_full['y'] == 'yes'][bank_additional_full.columns[i]] ,bank_additional_full[bank_additional_full['y'] == 'no'][bank_additional_full.columns[i]]],
                     10 ,rwidth=0.8 ,  color = ['g','r'] , label = ['yes','no'])
        else :
            plt.hist([bank_additional_full[bank_additional_full['y'] == 'yes'][bank_additional_full.columns[i]] ,bank_additional_full[bank_additional_full['y'] == 'no'][bank_additional_full.columns[i]]],
            rwidth=0.8 ,  color = ['g','r'] , label = ['yes','no'])
    ax = plt.subplot(4,5,20)
    plt.hist(bank_additional_full['y'])         
plt.subplots_adjust(wspace= .1 , hspace = .5 )
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/features stacked_histogram.png' , dpi = 300)

############################ Normalizing the numerical before categorical encoding #############################################

column_names_to_normalize = [i for i in bank_additional_full.columns if bank_additional_full[i].dtypes != 'O'  ]
for column in column_names_to_normalize:
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(np.array(bank_additional_full[column]).reshape(-1,1))
    bank_additional_full[column] = min_max_scaler.transform(np.array(bank_additional_full[column]).reshape(-1,1))
    bank_additional[column] = min_max_scaler.transform(np.array(bank_additional[column]).reshape(-1,1))    

##################################################### Encoding categorical variable  ######################################################################

    
if imputation :  # feature representation with imputing unknown variabl 
    train_imp = pd.DataFrame()
    test_imp = pd.DataFrame()
    trainset_sparse =  bank_additional_full.iloc[:,:-1]
    testset_sparse = bank_additional.iloc[:,:-1]
    mapping_values={}
    for column in columns[:-1]:
        if bank_additional_full[column].dtypes == 'O' :
            le = preprocessing.LabelEncoder()
            le.fit(bank_additional_full[column].unique())
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            mapping_values[column] = le_dict
            if 'unknown' in list(le_dict.keys()):
                missing_value = le_dict['unknown']
                a = le.transform(bank_additional_full[column])
                b = le.transform(bank_additional[column])
                # Another useful approach is using KNN from FancyImpute or regressing the missing value which is computationally expensive
                a = SimpleImputer(missing_values=missing_value, strategy='median').fit_transform(np.array(a).reshape(-1,1)).astype('int32')
                b = SimpleImputer(missing_values=missing_value, strategy='median').fit_transform(np.array(b).reshape(-1,1)).astype('int32')
                a = le.inverse_transform(a)
                b=le.inverse_transform(b)
                train_imp[column] = a
                test_imp[column] = b
            else:
                train_imp[column] = bank_additional_full[column]
                test_imp[column] = bank_additional[column]
            
        else:
            train_imp[column] = bank_additional_full[column]
            test_imp[column] = bank_additional[column]   
            
            
    encoder = ce.OneHotEncoder(cols=list(mapping_values.keys()))
    encoder.fit(train_imp)
    train = encoder.transform(train_imp)
    test = encoder.transform(test_imp)
    input_dimension = train.shape[1]
               

if hashing :
    hash_enc= ce.HashingEncoder()
    hash_enc.fit(bank_additional_full.iloc[:,:-1] , train_label )
    train = hash_enc.transform(bank_additional_full.iloc[:,:-1])
    test =  hash_enc.fit_transform(bank_additional.iloc[:,:-1])
    input_dimension = train.shape[1]

if onehot :#One_hot encoding without imputation , considering unknown a class for itself , # if col is not given all string would be sparsed 
    ohot_enc = ce.OneHotEncoder()
    ohot_enc.fit(bank_additional_full.iloc[:,:-1])
    train = ohot_enc.transform(bank_additional_full.iloc[:,:-1])
    test = ohot_enc.fit_transform(bank_additional.iloc[:,:-1])
    input_dimension = train.shape[1]
    #encoder.inverse_transform(testset_sparse[]) == testset could transfer back to real value 
    #encoder.mapping give the dictionary of all mappings  ==> encoder.mapping[0]


##################################################### Modeling  ######################################################################


#creat model builds a network basd on the parameters given
# dense_drop : list of tuples , each tupe (neuron , drop out rate )
#keras.regularizers.l1_l2(l1=0.01, l2=0.01)
#bias , zero or one ==> keras.initializers.Initializer() ,  keras.initializers.Ones() 
#optimizer from the choice here https://keras.io/optimizers/ 

def create_model( input_dimension =input_dimension  , dense_drop =[(4,0)]  , regularizer = regularizers.l2(0.01), bias ='Ones' , init_mode='uniform' , optimizer = 'adam' , loss = 'binary_crossentropy' ,  metrics=['accuracy'] ):
    # define model
    model = Sequential()
    for tup  in dense_drop: 
         model.add(Dense(tup[0], kernel_initializer=init_mode, activation='relu', 
                        input_dim=input_dimension , bias_initializer = bias , kernel_regularizer = regularizer  )) 
         model.add(Dropout(tup[1]))
    model.add(Dense(1, kernel_initializer=init_mode,bias_initializer = bias, kernel_regularizer = regularizer ))
    # compile model
    model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)
    return model



##################################################### Neural Net Greedy search  ######################################################################

start = time.time()  
greedy_search_model = KerasClassifier(build_fn=create_model, verbose=1)

Structure = [[(16,.0),(4,0)] , [(32,.01) , (16,0) , (4,0)] , [(8,0)] ]
Reg = [regularizers.l1_l2(l1=0.0, l2=0.01) , regularizers.l1_l2(l1=0.0, l2=0.00)]
Batches = [64,512 , 2024]
#Split_ratios=[.1 , .2] # because we are doing cross validation we dont need another validation set
#Weight = weight([1,5])   we dont need split ratio as we are doing kfold 


param_grid = dict(dense_drop = Structure , regularizer = Reg , batch_size = Batches )

grid = GridSearchCV(estimator=greedy_search_model, 
                    param_grid=param_grid,
                    cv=3)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
callbacks_list = [es]
grid_result = grid.fit(train, train_label ,epochs = 50, callbacks = callbacks_list)   # batch size should be large to have from the other distrubtion 
                                                                                            # with high probablity . However we take care of it later with 
duration = time.time() - start 
print('It takes {} minutes to search !!! Imagine if we involve more hyper parameters or we span our search ober each of them !! '.format(duration/60))                                                                                            # modifying the loss function , distribution change , 

print(f'Best Accuracy for hashing representation  {grid_result.best_score_:.4} using {grid_result.best_params_}')   # 
best_accuracy = grid_result.best_score_
best_network = grid_result.best_params_
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
greedy_CV = pd.concat([pd.Series(means) , pd.Series(stds) ] , axis = 1 )
greedy_CV.index = params # Show the highest score with lowest std , However cv = 3 might nit so valid                

##################################################### NN base model  ######################################################################

base_nn_model = create_model( dense_drop =[(16, 0.0), (4, 0)])
filepath = "/Users/hamedniakan/Desktop/Hamed/QuickenLoans/model_base_nn.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Graph_model_base_nn', histogram_freq=0,  
          write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
callbacks_list = [checkpoint, tbCallBack,es]

history_base_nn = base_nn_model.fit(np.array(train), train_label, epochs=50, 
                      batch_size=512, verbose=1 ,   callbacks=callbacks_list)                                                                             # It is all about the demo of NN structural hyper parameter tuning 


plt.figure()

ax = plt.subplot(1,2,1)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Loss')
plt.plot(history_base_nn.history['loss' ], label = 'train_loss' )
#plt.plot(history_base_nn.history['val_loss' ] , label = 'val_loss' )
plt.legend()
ax = plt.subplot(1,2,2)
plt.plot(history_base_nn.history['accuracy' ], label = 'train_accuracy' )
#plt.plot(history_base_nn.history['val_accuracy' ] , label = 'val_accuracy' )
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.title(' Accuracy')
plt.legend()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/acc_loss_base_nn.png' , dpi = 300)


prediction['base_nn'] = base_nn_model.predict_classes(test)


confusion_matrices['base_nn'] = confusion_matrix(y_true = test_label , y_pred= np.array(base_nn_model.predict_classes(test)))/test_label.shape[0]

precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(base_nn_model.predict_classes(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, base_nn_model.predict(test) )
roc_auc = auc(fpr, tpr)
 
print(confusion_matrices['base_nn'])


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic base_nn')
plt.legend(loc="lower right")
plt.show()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_base_nn.png' , dpi = 300)


Recall_Precision_Fmeasure_Auc['base_nn'] = [precision , recall , f , roc_auc]


positive_ratio = test_label[test_label==1].shape[0]/test_label.shape[0]

##################################################### Penalized NN  ######################################################################



penalized_nn_model = create_model( dense_drop =[(16, 0.0), (4, 0)])
#class_weight = compute_class_weight ('balanced' , np.unique(train_label) , train_label)
weight = np.where(train_label==0,1,7)
    
filepath = "/Users/hamedniakan/Desktop/Hamed/QuickenLoans/model_penalize_nn.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/GraphGraph_penalized', histogram_freq=0,  
          write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [checkpoint, tbCallBack,es]

history_penalized = penalized_nn_model.fit(np.array(train), train_label, epochs=100, 
                      batch_size=512, verbose=1,  validation_split = .1 , sample_weight = weight ,  callbacks=callbacks_list)


plt.figure()
plt.suptitle('Penalized NN')
ax = plt.subplot(1,2,1)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Loss')
plt.plot(history_penalized.history['loss' ], label = 'train_loss' )
plt.plot(history_penalized.history['val_loss' ] , label = 'val_loss' )
plt.legend()
ax = plt.subplot(1,2,2)
plt.plot(history_penalized.history['accuracy' ], label = 'train_accuracy' )
plt.plot(history_penalized.history['val_accuracy' ] , label = 'val_accuracy' )
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.title(' Accuracy')
plt.legend()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/acc_loss_penalized_nn.png' , dpi = 300)


prediction['penalized_nn'] = penalized_nn_model.predict_classes(test)

confusion_matrices['penalized_nn'] = confusion_matrix(y_true = test_label , y_pred= np.array(penalized_nn_model.predict_classes(test)))/test_label.shape[0]
precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(penalized_nn_model.predict_classes(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, penalized_nn_model.predict(test) )
roc_auc = auc(fpr, tpr)
 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Penalized NN')
plt.legend(loc="lower right")
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_penalized_nn.png' , dpi = 300)


Recall_Precision_Fmeasure_Auc['penalized_nn'] = [precision , recall , f , roc_auc]


##################################################### Resampling NN  ######################################################################

weight = np.where(train_label == 0 , 1 , 10) 

dist = weight/sum(weight)

Sample = np.random.choice(range(train_label.shape[0]) ,train_label.shape[0], p = dist)

X_train_resampling = np.array(train)[Sample]
Y_train_resampling = train_label[Sample]


resample_nn_model = create_model( dense_drop =[(16, 0.0), (4, 0)])
#class_weight = compute_class_weight ('balanced' , np.unique(train_label) , train_label)
#weight = np.where(train_label==0,1,9)
    
filepath = "/Users/hamedniakan/Desktop/Hamed/QuickenLoans/model_resample_nn.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph_resample', histogram_freq=0,  
          write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [checkpoint, tbCallBack,es]

history_resample = resample_nn_model.fit(np.array(X_train_resampling), Y_train_resampling, epochs=100, 
                      batch_size=512, validation_split = .2 ,verbose=1 ,  callbacks=callbacks_list)


plt.figure()
plt.suptitle('Resample NN')
ax = plt.subplot(1,2,1)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Loss')
plt.plot(history_resample.history['loss' ], label = 'train_loss' )
plt.plot(history_resample.history['val_loss' ] , label = 'val_loss' )
plt.legend()
ax = plt.subplot(1,2,2)
plt.plot(history_resample.history['accuracy' ], label = 'train_accuracy' )
plt.plot(history_resample.history['val_accuracy' ] , label = 'val_accuracy' )
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.title(' Accuracy')
plt.legend()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/acc_loss_Resample.png' , dpi = 300) # It s very good but it is not what reality is 

prediction['resample_nn'] = resample_nn_model.predict_classes(test)

confusion_matrices['resample_nn'] = confusion_matrix(y_true = test_label , y_pred= np.array(resample_nn_model.predict_classes(test)))/test_label.shape[0]
precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(resample_nn_model.predict_classes(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, resample_nn_model.predict(test) )
roc_auc = auc(fpr, tpr)
 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for resample NN')
plt.legend(loc="lower right")
plt.show()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_Resample.png' , dpi = 300)


Recall_Precision_Fmeasure_Auc['resample_nn'] = [precision , recall , f , roc_auc]

##################################################### Upsampling NN NN  ######################################################################
neg_szie = train_label[train_label==0].shape[0]
train_data = pd.concat([train, pd.Series(train_label)] , axis =1)
neg_data = train_data[train_data[0]==0]
pos_data = train_data[train_data[0]==1]

pos_upsampled = resample(pos_data, 
                                 replace=True,     # sample with replacement
                                 n_samples=neg_szie,    # to match majority class
                                 random_state=123) # reproducible results


train_upsample = pd.concat([neg_data,pos_upsampled ])
train_upsample = shuffle(train_upsample)

X_train = np.array(train_upsample.iloc[:,:-1])
y_train = np.array(train_upsample.iloc[:,-1])

upsample_nn_model = create_model( dense_drop =[(16, 0.0), (4, 0)])
#class_weight = compute_class_weight ('balanced' , np.unique(train_label) , train_label)
#weight = np.where(train_label==0,1,9)
    
filepath = "/Users/hamedniakan/Desktop/Hamed/QuickenLoans/model_upsample_nn.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Graph_upsample', histogram_freq=0,  
          write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [checkpoint, tbCallBack,es]

history_upsample = upsample_nn_model.fit(np.array(X_train), y_train, epochs=100, 
                      validation_split = .2 ,batch_size=512, verbose=1 ,  callbacks=callbacks_list)

plt.figure()
plt.suptitle('upsampling NN')
ax = plt.subplot(1,2,1)
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Loss')
plt.plot(history_upsample.history['loss' ], label = 'train_loss' )
plt.plot(history_upsample.history['val_loss' ] , label = 'val_loss' )
plt.legend()
ax = plt.subplot(1,2,2)
plt.plot(history_upsample.history['accuracy' ], label = 'train_accuracy' )
plt.plot(history_upsample.history['val_accuracy' ] , label = 'val_accuracy' )
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.title(' Accuracy')
plt.legend()
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/acc_loss_UPsample.png' , dpi = 300) # It s very good but it is not what reality is 


prediction['upsample_nn'] = upsample_nn_model.predict_classes(test)

confusion_matrices['upsample_nn'] = confusion_matrix(y_true = test_label , y_pred= np.array(upsample_nn_model.predict_classes(test)))/test_label.shape[0]
precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(upsample_nn_model.predict_classes(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, upsample_nn_model.predict(test) )
roc_auc = auc(fpr, tpr)
 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for upsample NN')
plt.legend(loc="lower right")
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_UPsample.png' , dpi = 300)


Recall_Precision_Fmeasure_Auc['upsample_nn'] = [precision , recall , f , roc_auc]

##################################################### Random Forest  ######################################################################

R_squred ={}

RF = RandomForestClassifier(n_estimators=100, max_depth=2,
                            random_state=0 , class_weight = {0:1,1:10} , bootstrap = True)
RF.fit(train, train_label)

prediction['Random forest'] = RF.predict(test)

confusion_matrices['Random Forest'] = confusion_matrix(y_true = test_label , y_pred= np.array(RF.predict(test)))/test_label.shape[0]
precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(RF.predict(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, RF.predict_proba(test)[:,1] )
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Random Forest')
plt.legend(loc="lower right")
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_RF.png' , dpi = 300)


importance = RF.feature_importances_
indices_tree = np.argsort(RF.feature_importances_)[::-1]

fig= plt.figure(figsize=(12,8))
plt.bar(range(20) , importance[indices_tree[:20]])
plt.xticks(range(20), indices_tree[:20])
plt.xlabel('Feature' , fontsize = labelfont_size)
plt.ylabel('Importance', fontsize = labelfont_size)
plt.tick_params(labelsize=ticksize, pad=6)
plt.table(cellText=np.array(pd.DataFrame(train.columns[indices_tree[:20]] ,index = indices_tree[:20])) ,
          cellLoc = ['right','left'], colWidths = [.2,.2] , loc = 'upper center' , rowLabels = indices_tree[:20] , edges = 'open' )
plt.text ( .5,.2, 'R^2 = {:.2f}'.format(RF.score(test , test_label)) , style = 'italic' , fontsize = 14 , fontweight = 'bold' )
fig.suptitle('Random Forest Feature Importance Analysis' , fontweight = 'bold' , y=0.94, fontsize = figfont_size)
fig.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/Feature_importance_RF.png' , dpi = 300)


R_squred['RF'] = RF.score(test , test_label)
Recall_Precision_Fmeasure_Auc['RF'] = [precision , recall , f , roc_auc]

##################################################### XGBOOST   ######################################################################



xgb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=.01, max_features="log2", max_depth=5, 
                                       random_state=0 , verbose = 1)
xgb_model.fit(train, train_label , sample_weight=weight )   
 
    
confusion_matrices['XGboost'] = confusion_matrix(y_true = test_label , y_pred= np.array(xgb_model.predict(test)))/test_label.shape[0]
prediction['XGBoost'] = xgb_model.predict(test)  

precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(xgb_model.predict(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, xgb_model.predict_proba(test)[:,1] )
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for XGBoost')
plt.legend(loc="lower right")
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_Xgboost.png' , dpi = 300)


importance = xgb_model.feature_importances_
indices_tree = np.argsort(xgb_model.feature_importances_)[::-1]

fig= plt.figure(figsize=(12,8))
plt.bar(range(20) , importance[indices_tree[:20]])
plt.xticks(range(20), indices_tree[:20])
plt.xlabel('Feature' , fontsize = labelfont_size)
plt.ylabel('Importance', fontsize = labelfont_size)
plt.tick_params(labelsize=ticksize, pad=6)
plt.table(cellText=np.array(pd.DataFrame(train.columns[indices_tree[:20]] ,index = indices_tree[:20])) ,
          cellLoc = ['right','left'], colWidths = [.2,.2] , loc = 'upper center' , rowLabels = indices_tree[:20] , edges = 'open' )
plt.text ( .5,.2, 'R^2 = {:.2f}'.format(xgb_model.score(test , test_label)) , style = 'italic' , fontsize = 14 , fontweight = 'bold' )
fig.suptitle('Xgboost Feature Importance Analysis' , fontweight = 'bold' , y=0.94, fontsize = figfont_size)
fig.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/Feature_importance_XGboost.png' , dpi = 300)


R_squred['XGboost'] = RF.score(test , test_label)  
Recall_Precision_Fmeasure_Auc['XGBoost'] = [precision , recall , f , roc_auc]
    

LR = LogisticRegression(class_weight = {0:1,1:10} )
LR.fit(train , train_label)    

confusion_matrices['Log Reg'] = confusion_matrix(y_true = test_label , y_pred= np.array(LR.predict(test)))/test_label.shape[0]
prediction['log Reg'] = LR.predict(test)  

precision , recall , f , _ = prf(y_true = test_label , y_pred= np.array(LR.predict(test)) , average = 'binary')

fpr, tpr, _ = roc_curve(test_label, LR.predict_proba(test)[:,1] )
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Lasso Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/AUC_LR.png' , dpi = 300)
 
##################################################### Logistic Regression  ######################################################################

importance = LR.coef_[0,:]
indices_coef = np.argsort(LR.coef_)[0,:][::-1]

fig= plt.figure(figsize=(12,8))
plt.bar(range(20) , importance[indices_coef[:20]])
plt.xticks(range(20), indices_coef[:20])
plt.xlabel('Feature' , fontsize = labelfont_size)
plt.ylabel('Importance', fontsize = labelfont_size)
plt.tick_params(labelsize=ticksize, pad=6)
plt.table(cellText=np.array(pd.DataFrame(train.columns[indices_coef[:20]] ,index = indices_coef[:20])) ,
          cellLoc = ['right','left'], colWidths = [.2,.2] , loc = 'upper center' , rowLabels = indices_coef[:20] , edges = 'open' )
plt.text ( .5,.8, 'R^2 = {:.2f}'.format(LR.score(test , test_label)) , style = 'italic' , fontsize = 14 , fontweight = 'bold' )
fig.suptitle('Logistic Regression Feature Importance Analysis' , fontweight = 'bold' , y=0.94, fontsize = figfont_size)
fig.savefig('/Users/hamedniakan/Desktop/Hamed/QuickenLoans/Feature_importance_LR.png' , dpi = 300)


R_squred['Log Reg'] = LR.score(test , test_label)  
Recall_Precision_Fmeasure_Auc['Log reg '] = [precision , recall , f , roc_auc]


        
#####################################################Conclusion  ######################################################################
    
model_performance = pd.DataFrame.from_dict(Recall_Precision_Fmeasure_Auc , orient  = 'index' , columns = ['Precision','Recall','F_measure' , 'AUC'])    
    




