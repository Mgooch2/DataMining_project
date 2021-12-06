#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:13:08 2021

@author: mionnegooch
"""
# CS4412 : Data Mining
# Fall 2021
# Kennesaw State University
# Extra Credit Project

"""In this project, I explored a Dungeons and Dragons dataset available at :https://github.com/oganm/dnddata

This dataset is a collection of over 7000 player character sheets submitted on the creators web apps.
This project will focus on exploring player character choices.
"""
#%%
#import modules
import pandas as pd
import matplotlib.pyplot as plt

#%%
# import dataset
dataset_fn = "dnd_chars_unique.csv" 
#%%
# read dataset in by column
col_list = ["background","justClass","level","HP","AC","processedRace","processedWeapons"]
data = pd.read_csv(dataset_fn,usecols=col_list)

#%%
# split columns with multiple data
data['processedWeapons'] = data['processedWeapons'].str.split('[|]')
data = data.explode('processedWeapons').reset_index(drop=True)
data['justClass'] = data['justClass'].str.split('[|]')
data = data.explode('justClass').reset_index(drop=True)

cols = list(data.columns)
data = data[cols]

#%%

j_class = data['justClass'].value_counts().rename_axis('Feature').reset_index(name='Counts')
p_race = data['processedRace'].value_counts().rename_axis('Feature').reset_index(name='Counts')
bg = data['background'].value_counts().rename_axis('Feature').reset_index(name='Counts')
p_weapons = data['processedWeapons'].value_counts().rename_axis('Feature').reset_index(name='Counts')

        
#%%
#create pie charts
def pie_chart(x):
    plt.pie(x['Counts'], labels= x['Feature'], shadow= True,autopct='%1.1f%%')
    plt.axis('equal')
    plt.title('Feature')
    plt.show()

#%%
bk = 'Background'
w = 'Weapons'
def vert_bar(x,y):
    b = plt.bar(x['Feature'][0:5],x['Counts'][0:5])
    plt.bar_label(b, fmt='%.0f')
    plt.title('Top 5' + y)
    plt.xlabel('Counts')
    plt.ylabel('Feature: ' + y)
    plt.show()
def horz_bar(x,y):
    b = plt.barh(x['Feature'][0:10],x['Counts'][0:10])
    plt.bar_label(b, fmt='%.0f')
    plt.title('Top 10' + y)
    plt.xlabel('Counts')
    plt.ylabel('Feature: ' + y)
    plt.show()



