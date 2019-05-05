# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:27:59 2019

@author: Abhi
"""

import numpy as np
import pandas as pd
import json
import os

def readJson(x):
    with open(x, "r", encoding="utf8") as read_file:
        data = json.load(read_file)
    return(data)

def flatten_json(nested_json):
    """
        Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

SentDir = "C:/Users/Abhi/Documents/Data Science Lab/Metadata"
os.chdir(SentDir)
files = os.listdir()
files = [x for x in files if 'json' in x]
files
for i in files:
    
    df = readJson(i)
    df = flatten_json(df)
    
    a = pd.DataFrame.from_dict(df, orient='index')
    a.columns = ['Values']
    
    label = a[a.index.str.match('label')]
    
    if label.shape[0]>0:
        label.index = label.index.str.split('_',expand=True)
        label.reset_index(inplace=True)
        label.drop(columns=['level_0'], inplace=True)
        label = label[label.level_2.str.match('score')]
        label.sort_values(by=['Values'], ascending=[0], inplace=True)
        #label['level_2'] = label['level_2'] + '_' + label['level_1'].astype(str)
        #label.drop(columns=['level_1'], inplace=True)
        label.reset_index(inplace=True)
        label.drop(columns=['index'], inplace=True)
      
        #label.set_index('index', inplace=True)
        #label.reset_index(inplace=True, drop=True)
        #label = pd.pivot_table(label, index='level_1', columns='level_2', values='Values',aggfunc='first' ).reset_index()
    
        if label.shape[0] > 4:
            label = label.iloc[:5,:]
            
        
        #label['level_1'] = label.index
        label['level_1'] = label['level_2'] + '_' + label['level_1'].astype(str)
        label.drop(columns=['level_2'], inplace = True)
        label = pd.melt(label, id_vars=['level_1'])
        label.drop(columns=['variable'], inplace = True)
        label.set_index('level_1', inplace=True)
        label = label.transpose()
        label.reset_index(inplace=True, drop=True)
    
    image = a[a.index.str.match('image')]
    
    if image.shape[0]>0:
        image.index = image.index.str.split('_',expand=True)
        image.reset_index(inplace=True)        
        #image = image.groupby('level_5')
        image.reset_index(inplace=True)
        image.drop(columns= ['index','level_0','level_1','level_2'], inplace=True)
        #image['level_4'] = image['level_4'] + '_' + image['level_5'].astype(str)
        image['level_4'] = image['level_4'] + '_' + image['level_5'].astype(str)
        image.drop(columns=['level_5'], inplace=True)
        image = pd.pivot_table(image, index='level_3', columns='level_4', values='Values',aggfunc='first' ).reset_index()
        #image['level_3'] = image.index
        #image.set_index('level_3', inplace=True)
                
        if image.shape[0] > 4:
            image = image.iloc[:5,:]
        #eimagentities.drop(columns=['type'], inplace=True)
        image = pd.melt(image, id_vars=['level_3'])
        image['level_3'] = image['level_4'] + '_' + image['level_3'].astype(str)
        image.drop(columns=['level_4'], inplace=True)
        image.set_index('level_3', inplace=True)
        image = image.transpose()
        image.reset_index(inplace=True, drop=True)
    
    crop = a[a.index.str.match('crop')]
    
    if crop.shape[0]>0:
        crop.index = crop.index.str.split('_',expand=True)
        crop.reset_index(inplace=True)
        crop.drop(columns=['level_0','level_1', 'level_2'], inplace=True)
        #crop = crop.groupby('level_6')
        crop['level_3'] = crop['level_3'] + '_' + crop['level_6'].astype(str)
        crop.drop(columns=['level_4'], inplace=True)
        crop.drop(columns=['level_6'], inplace=True)
       
        bounding = crop[crop.level_3.str.match('bounding')]
        if bounding.shape[0] > 0:
            bounding['level_3'] = bounding['level_3'] + '_' + bounding['level_5']
            bounding.drop(columns=['level_5'], inplace=True)
            bounding.set_index('level_3', inplace=True)
            bounding = bounding.transpose()
            bounding.reset_index(inplace=True, drop=True)
        
        
        confidence = crop[crop.level_3.str.match('confidence')]
        if crop.shape[0]>0:
            confidence = pd.DataFrame(confidence)
            confidence = confidence['Values']
            confidence = pd.DataFrame(confidence)
            confidence.columns = np.array(['Confidence'])
            confidence.reset_index(inplace=True, drop=True)

        importance = crop[crop.level_3.str.match('importance')]
     
        if importance.shape[0]>0 :
            importance = crop[crop.level_3.str.match('importance')]
            importance = importance['Values'] 
            importance.reset_index(inplace=True, drop=True)
            importance = pd.DataFrame(importance)
            importance.columns = np.array(['Importance'])
            
    crophints = pd.concat([bounding,confidence,importance], axis=1)
        
        #crop['level_3'] = crop['level_3'] + '_' + crop['level_5'].astype(str)
        #crop.drop(columns=['level_5'], inplace=True)
        #crop = pd.pivot_table(crop, index='level_5', columns='level_3', values='Values',aggfunc='first' ).reset_index()
        #crop = pd.melt(crop, id_vars=['level_3'])
        #crop = pd.pivot_table(crop, index='level_5', columns='level_3', values='Values',aggfunc='first' ).reset_index()
        #crop.sort_values(by=['score'], ascending=[0], inplace=True)
        #crop.reset_index(inplace=True, drop=True)
    
    final = pd.concat([label,image, crophints], axis=1)
    
    if i == files[0]:
        Result = final
    else:
        Result = pd.concat([Result, final])

Result.to_csv("Metadata.csv")