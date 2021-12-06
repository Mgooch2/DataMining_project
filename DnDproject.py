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
This project will focus on exploring player character choices, specifically race, background, class, and weapon choice.
"""
#%%
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as pltimg


#%%
# import dataset
dataset_fn = "dnd_chars_unique.csv" 
#%%
# read dataset in by column
col_list = ["background","justClass","processedRace","processedWeapons"]
data = pd.read_csv(dataset_fn,usecols=col_list)

#%%
# split columns with multiple data
data['justClass'] = data['justClass'].str.split('[|]')
data = data.explode('justClass').reset_index(drop=True)
data['processedWeapons'] = data['processedWeapons'].str.split('[|]')
data = data.explode('processedWeapons').reset_index(drop=True)
# put columns into a list
#cols = list(data.columns)
#data = data[cols]
#print(data)
#%%
# takes each column and splits into mini table of features and the count of each item in dataset
j_class = data['justClass'].value_counts().rename_axis('Class').reset_index(name='Counts')
p_race = data['processedRace'].value_counts().rename_axis('Race').reset_index(name='Counts')
bg = data['background'].value_counts().rename_axis('Background').reset_index(name='Counts')
p_weapons = data['processedWeapons'].value_counts().rename_axis('Weapon').reset_index(name='Counts')


#%%
# for chart labels
bk = 'Background'
w = 'Weapon'
r = 'Race'
cl = 'Class'        
#%%
#function to create pie charts
def pie_chart(x,y):
    plt.pie(x['Counts'], labels= x[y], shadow= True,autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(y)
    plt.show()

#%%

# functions for vertical and horizonal bar graphs
def vert_bar(x,y):
    b = plt.bar(x[y][0:5],x['Counts'][0:5])
    plt.bar_label(b, fmt='%.0f')
    plt.title('Top 5 ' + y + ' Selections')
    plt.xlabel('Counts')
    plt.ylabel(y)
    plt.show()
def horz_bar(x,y):
    b = plt.barh(x[y][0:10],x['Counts'][0:10])
    plt.bar_label(b, fmt='%.0f')
    plt.title('Top 10 ' + y + ' Selections')
    plt.xlabel('Counts')
    plt.ylabel(y)
    plt.show()
#%%
# create a decision tree

# add new column for data for create character for particular attribute
data['Create'] = np.where(data['processedWeapons'] == 'Longbow', 'Yes','No')

#convert all data to numerical values
d = {'Acolyte':0,'Charlatan':1,'Criminal':2,'Entertainer':3,'Folk Hero':4,
     'Guild Artisan':5,'Hermit':6,'Noble':7,'Outlander':8,'Sage':9,
     'Sailor':10,'Soldier':11,'Urchin':12,'Pirate':13,'Knight':14,
     'Guild Merchant':15,'Gladiator':16,'Spy':17}
data['background'] = data['background'].map(d)
data['background'].fillna(18, inplace=True) #replace NaN with number
#races
d1 = {'Dwarf':0,'Elf':1,'High Elf':2,'Wood Elf':3,'Dark Elf':4,
     'Halfling':5,'Human':6,'Dragonborn':7,'Gnome':8,
     'Half-Elf':9, 'Half-Orc':10, 'Tiefling':11, 'Tabaxi':12, 
     'Aasimar':13, 'Genasi':14,'Firbolg':15,'Goliath':16,
     'Turtle':17,"Warforged":18,'Custom':19}
data['processedRace'] = data['processedRace'].map(d1)
data['processedRace'].fillna(20,inplace=True)#replace NaN with number
#classes
d2 = {'Barbarian':0,'Bard':1,'Cleric':2,'Druid':3,'Fighter':4,
      'Monk':5,'Paladin':6,'Ranger':7,'Rogue':8,'Sorcerer':9,
      'Warlock':10, 'Wizard':11,'Artificer':12}
data['justClass'] = data['justClass'].map(d2)
data['justClass'].fillna(13,inplace=True)#replace NaN with number
#weapons
d3 = {'Club':0,'Dagger':1,'Greatclub':2,'Handaxe':3,'Javelin':4,
      'Light Hammer':5,'Mace':6,'Quarterstaff':7,'Sickle':8,'Spear':9,
      'Crossbow, Light':10,'Dart':11,'Shortbow':12,'Sling':13,'Battleaxe':14,
      'Flail':15,'Glaive':16,'Greataxe':17,'Greatsword':18,'Halberd':19,
      'Lance':20,'Longsword':21,'Maul':22,'Morningstar':23,'Pike':24,
      'Rapier':25,'Scimitar':26,'Shortsword':27,'Trident':28,'War Pick':29,
      'Warhammer':30,'Whip':31,'Blowgun':32,'Crossbow, Hand':33,'Crossbow, Heavy':34,
      'Longbow':35,'Net':36,'Unarmed Strike':37}
data['processedWeapons'] = data['processedWeapons'].map(d3)
data['processedWeapons'].fillna(38,inplace=True)#replace NaN with number
#create character
d4  = {'Yes':1,'No':0}
data['Create'] = data['Create'].map(d4)

#seperate features columns form target column
features = ['background', 'justClass', 'processedRace', 'processedWeapons']
x = data[features]
y = data['Create']
# function to create tree
def d_tree(x,y):
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(x, y)
    data_tree = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data_tree)
    graph.write_png('mydecisiontree.png')
    img=pltimg.imread('mydecisiontree.png')
    #imgplot = plt.imshow(img)
    plt.show()


