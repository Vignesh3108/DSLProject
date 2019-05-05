# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:37:15 2019

@author: bhvig
"""
import cv2
import pandas as pd
import numpy as np
import os
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Input, Dropout
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping

# Set Working Directory
os.chdir('D:/Users/srinj/IIM-C Drive/Semester 2/Data Science Lab/Project')

# # =============================================================================
# # import warnings
# # warnings.filterwarnings("ignore")
# # =============================================================================

# ####### Section 1: SentimentJson to CSV

# SentDir = "Data/train_sentiment"
# files = os.listdir(SentDir)
# files = [x for x in files if 'json' in x]

# def readJson(x):
    # with open(x, "r", encoding="utf8") as read_file:
        # data = json.load(read_file)
    # return(data)

# def flatten_json(nested_json):
    # """
        # Flatten json object with nested keys into a single level.
        # Args:
            # nested_json: A nested json object.
        # Returns:
            # The flattened json object if successful, None otherwise.
    # """
    # out = {}

    # def flatten(x, name=''):
        # if type(x) is dict:
            # for a in x:
                # flatten(x[a], name + a + '_')
        # elif type(x) is list:
            # i = 0
            # for a in x:
                # flatten(a, name + str(i) + '_')
                # i += 1
        # else:
            # out[name[:-1]] = x

    # flatten(nested_json)
    # return out

# def SentFunct(i):
    # df = readJson(i)
    # df = flatten_json(df)
    
    # a = pd.DataFrame.from_dict(df, orient='index')
    # a.columns = ['Values']
    
    # sentences = a[a.index.str.match('sentences')]
    
    # if sentences.shape[0]>0:
        # sentences.index = sentences.index.str.split('_',expand=True)
        # sentences.reset_index(inplace=True)
        # sentences.drop(columns=['level_0', 'level_2'], inplace=True)
        # sentences = pd.pivot_table(sentences, index='level_1', columns='level_3', values='Values',aggfunc='first' ).reset_index()
        # sentences.sort_values(by=['magnitude'], ascending=[0], inplace=True)
        # sentences.reset_index(inplace=True, drop=True)
        # if sentences.shape[0] > 4:
            # sentences = sentences.iloc[:5,:]
        # sentences.drop(columns=['level_1' ,'beginOffset', 'content', 'magnitude'], inplace=True)
        # sentences['level_1'] = sentences.index
        # sentences.set_index('level_1', inplace=True)
        # sentences = sentences.transpose()
        # sentences = sentences.add_prefix('sentences_')
        # sentences.reset_index(inplace=True, drop=True)
    
    # entities = a[a.index.str.match('entities')]
    
    # if entities.shape[0]>0:
        # entities.index = entities.index.str.split('_',expand=True)
        # entities.reset_index(inplace=True)
        # mentions = entities[entities['level_2']=='mentions']
        # mentions = pd.DataFrame(mentions.groupby('level_1')['level_3'].max())
        # mentions.reset_index(inplace=True)
        # mentions.columns = ['level_1', "Mentions"]
        # mentions["Mentions"] = pd.to_numeric(mentions["Mentions"]) + 1
        # entities = entities[entities['level_2']!='mentions']
        # entities.drop(columns=['level_0', 'level_3', 'level_4', 'level_5'], inplace=True)
        # entities = pd.pivot_table(entities, index='level_1', columns='level_2', values='Values',aggfunc='first' ).reset_index()
        # entities = pd.merge(entities, mentions, how="inner", on='level_1')
        # entities.sort_values(by=['Mentions', 'salience'], ascending=[0, 0], inplace=True)
        # entities.reset_index(inplace=True, drop=True)
        # entities['level_1'] = entities.index
        # if entities.shape[0] > 4:
            # entities = entities.iloc[:5,:]
        # entities.drop(columns=['type'], inplace=True)
        # entities = pd.melt(entities, id_vars=['level_1'])
        # entities['level_1'] = entities['variable'] + '_' + entities['level_1'].astype(str)
        # entities.drop(columns=['variable'], inplace=True)
        # entities.set_index('level_1', inplace=True)
        # entities = entities.transpose()
        # entities.reset_index(inplace=True, drop=True)
    
    # document = a[a.index.str.match('document')]
    # if document.shape[0]>0:
        # document = document.transpose()
        # document.reset_index(inplace=True, drop=True)
        # document['PetID'] = (i.split('.')[0]).split('/')[2]
    
    # language = a[a.index.str.match('language')]
    # if language.shape[0]>0:
        # language = language.transpose()
        # language.reset_index(inplace=True, drop=True)
    
    # final = pd.concat([language, document, sentences, entities], axis=1)
    # return(final)
      
# TrainSentiment = Parallel(n_jobs=-1)(delayed(SentFunct)(SentDir + '/' + i) for i in files)
# TrainSentiment = pd.concat(TrainSentiment, sort=False)
# TrainSentiment = TrainSentiment.filter(regex= 'Mention*|PetID*|document*|lang*|sentences*')

# TrainSentiment.to_csv('Results/TrainSentiment.csv')

# ####### Section 2: MetaDataJson to CSV

# MetaDir = "Data/train_metadata"
# files = os.listdir(MetaDir)
# files = [x for x in files if 'json' in x]

# def MetaFunct(i):  
    # df = readJson(i)
    # df = flatten_json(df)
    
    # a = pd.DataFrame.from_dict(df, orient='index')
    # a.columns = ['Values']
    
    # label = a[a.index.str.match('label')]
    
    # if label.shape[0]>0:
        # label.index = label.index.str.split('_',expand=True)
        # label.reset_index(inplace=True)
        # label.drop(columns=['level_0'], inplace=True)
        # label = label[label.level_2.str.match('score')]
        # label.sort_values(by=['Values'], ascending=[0], inplace=True)
        # label.reset_index(inplace=True)
        # label.drop(columns=['index'], inplace=True)
   
        # if label.shape[0] > 4:
            # label = label.iloc[:5,:]
        
        # label['level_1'] = label['level_2'] + '_' + label['level_1'].astype(str)
        # label.drop(columns=['level_2'], inplace = True)
        # label = pd.melt(label, id_vars=['level_1'])
        # label.drop(columns=['variable'], inplace = True)
        # label.set_index('level_1', inplace=True)
        # label = label.transpose()
        # label.reset_index(inplace=True, drop=True)
    
    # image = a[a.index.str.match('image')]
    
    # if image.shape[0]>0:
        # image.index = image.index.str.split('_',expand=True)
        # image.reset_index(inplace=True)        
        # image.reset_index(inplace=True)
        # image.drop(columns= ['index','level_0','level_1','level_2'], inplace=True)
        # image['level_4'] = image['level_4'] + '_' + image['level_5'].astype(str)
        # image.drop(columns=['level_5'], inplace=True)
        # image = pd.pivot_table(image, index='level_3', columns='level_4', values='Values',aggfunc='first' ).reset_index()
                
        # if image.shape[0] > 4:
            # image = image.iloc[:5,:]
        # image = pd.melt(image, id_vars=['level_3'])
        # image['level_3'] = image['level_4'] + '_' + image['level_3'].astype(str)
        # image.drop(columns=['level_4'], inplace=True)
        # image.set_index('level_3', inplace=True)
        # image = image.transpose()
        # image.reset_index(inplace=True, drop=True)
    
    # crop = a[a.index.str.match('crop')]
    
    # if crop.shape[0]>0:
        # crop.index = crop.index.str.split('_',expand=True)
        # crop.reset_index(inplace=True)
        # crop.drop(columns=['level_0','level_1', 'level_2'], inplace=True)
        # crop['level_3'] = crop['level_3'] + '_' + crop['level_6'].astype(str)
        # crop.drop(columns=['level_4'], inplace=True)
        # crop.drop(columns=['level_6'], inplace=True)
       
        # bounding = crop[crop.level_3.str.match('bounding')]
        # if bounding.shape[0] > 0:
            # bounding['level_3'] = bounding['level_3'] + '_' + bounding['level_5']
            # bounding.drop(columns=['level_5'], inplace=True)
            # bounding.set_index('level_3', inplace=True)
            # bounding = bounding.transpose()
            # bounding.reset_index(inplace=True, drop=True)
        
        
        # confidence = crop[crop.level_3.str.match('confidence')]
        # if crop.shape[0]>0:
            # confidence = pd.DataFrame(confidence)
            # confidence = confidence['Values']
            # confidence = pd.DataFrame(confidence)
            # confidence.columns = np.array(['Confidence'])
            # confidence.reset_index(inplace=True, drop=True)

        # importance = crop[crop.level_3.str.match('importance')]
     
        # if importance.shape[0]>0 :
            # importance = crop[crop.level_3.str.match('importance')]
            # importance = importance['Values'] 
            # importance.reset_index(inplace=True, drop=True)
            # importance = pd.DataFrame(importance)
            # importance.columns = np.array(['Importance'])
            
    # crophints = pd.concat([bounding,confidence,importance], axis=1)
    # print(i)
    # document= pd.DataFrame({"PetID":[(i.split('.')[0]).split('/')[2]]})
    # final = pd.concat([document, label, image, crophints], axis=1)
    # return(final)
    
# TrainMetaData = Parallel(n_jobs=-1)(delayed(MetaFunct)(MetaDir + '/' + i) for i in files)
# TrainMetaData = pd.concat(TrainMetaData)

# TrainMetaData.to_csv('Results/TrainMetaData.csv')
###############Dont run Always
TrainMetaData = pd.read_csv('Results/TrainMetaData.csv')
TrainSentiment = pd.read_csv('Results/TrainSentiment.csv')


####### Section 3: Image resizing

data_dir = 'Data'

def resize_to_square(im, img_size=224):
    old_size = im.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])    
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im
  
def load_image(filename, im_sz=224):
    imageee = cv2.imread(filename)
    new_image = resize_to_square(imageee, img_size=im_sz)
    return new_image

def ImageLoad(i):
    img = load_image(i, im_sz=224)
    Petid = i.split('.')[0]
    Petid = Petid.split('/')[2]
    return img, Petid

images = os.listdir(data_dir + '/train_images/')
images = [i for i in images if 'jpg' in i]
imgs = Parallel(n_jobs=-1)(delayed(ImageLoad)(data_dir + '/train_images/' + i) for i in images)
imgs = pd.DataFrame(imgs)
imgs.columns = np.array(['ndarray', 'PetId'])
img = list(imgs['ndarray'])

img = np.array(img)
PetID = imgs['PetId']

####### Section 4: Reordering the Data Set
Data_in = pd.read_csv('Data/train.csv')
Data1 = Data_in[[ 'Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee', 'PetID', 'PhotoAmt', 'AdoptionSpeed', 'VideoAmt',]]
Data2 = Data_in[['Type', 'Breed1', 'Breed2',  'Gender', 'Color1', 'Color2', 'Color3', 'Vaccinated', 'Dewormed','Sterilized', 'Health', 'State']]
Data2 = pd.get_dummies(Data2.astype('category'))

Data_in = pd.concat([Data1, Data2], axis=1)

Data_in=Data_in.sort_values('PetID')
MergeSkel = pd.DataFrame(PetID)
MergeSkel.columns = ["Key"]
Mer=MergeSkel['Key'].str.split('-', n = 1, expand = True)
MergeSkel['PetId']=Mer[0]
MergeSkel['PNo']=Mer[1]

lang = TrainSentiment.pop('language')
lang = pd.get_dummies(lang)
TrainSentiment = pd.concat([TrainSentiment, lang], axis=1)

MergeSkel = pd.merge(MergeSkel, TrainMetaData, how='inner', left_on='Key', right_on='PetID' )
Data_in = pd.merge(Data_in, TrainSentiment, how='outer', left_on='PetID', right_on='PetID')
Data_in = pd.merge(MergeSkel, Data_in, how='inner', left_on='PetId', right_on="PetID")

# Reorder
Order = PetID.argsort()
PetID = PetID[Order]
img = img[Order]
Data_in = Data_in.sort_values('Key')

#Final Go Ahead
Data_in['Key'].equals(PetID)
Data_in.to_csv('Results/Data_in.csv')

a = Data_in.isna().sum()
Data_in.drop(["Values", "boundingPoly_x_0", "boundingPoly_x_3", "boundingPoly_y_0", "boundingPoly_y_1", "level_3", "level_5"], axis=1, inplace=True)
Data_in.fillna(0, inplace=True)
Data_in.info()
Data_in.drop(['Key', 'PetId', 'PNo' , 'PetID_x', 'PetID_y', ], axis=1, inplace=True)
Y_train = np.array(Data_in['AdoptionSpeed'])
Y_train = pd.get_dummies(Y_train)
Data_in.drop(['AdoptionSpeed'], axis=1, inplace=True)
X_Train = np.array(Data_in)



####### Section 5: Building the neural Network
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
percent = 0 # percent of trainable layers
    
for layer in base_model.layers:
    layer.trainable = False

# add a global spatial average pooling layer to the inception model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)
base_model_modified = Model(inputs=base_model.input, outputs=x)

# let's add a fully-connected layer whi
inputs = Input(shape=(423,))
y = Dense(423, activation='relu')(inputs)
#y = Dense(423, activation='relu')(y)
y = Dense(50, activation='relu')(y)
y = Model(inputs=inputs, outputs=y)

# and a logistic layer -- let's say we have 200 classes
combined = concatenate([base_model_modified.output, y.output])
z = Dense(512, activation="relu")(combined)
z = Dropout(0.4)(z)
#z = Dense(512, activation="relu")(z)
#z = Dropout(0.4)(z)
#z = Dense(512, activation="relu")(z)
predictions = Dense(5, activation='softmax')(z)

# this is the model we will train
model = Model(inputs=[base_model_modified.input, y.input], outputs=predictions)
model.summary()

# Define early_stopping_monitor
#early_stopping_monitor = EarlyStopping(patience=5)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit([img, X_Train], Y_train, validation_split=0.2, batch_size=100, epochs=10)

for layer in model.layers[280::]:
    layer.trainable = True

model.fit([img, X_Train], Y_train, validation_split=0.2, batch_size=100, epochs=10)

model.predict([img[150:155], X_Train[150:155,:]])
