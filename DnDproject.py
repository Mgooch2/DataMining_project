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
bg = data['background'].value_counts().rename_axis('Background').reset_index(name='Counts')
j_class = data['justClass'].value_counts().rename_axis('Class_name').reset_index(name='Counts')
p_race = data['processedRace'].value_counts().rename_axis('Race').reset_index(name='Counts')
p_weapons = data['processedWeapons'].value_counts().rename_axis('Weapons').reset_index(name='Counts')
print()
        
#%%
plt.pie(j_class['Counts'], labels=j_class['Class_name'], shadow= True,autopct='%1.1f%%')
plt.axis('equal')
plt.title('Player Classes')
plt.show()

plt.pie(p_race['Counts'], labels=p_race['Race'], shadow= True,autopct='%1.1f%%')
plt.axis('equal')
plt.title('Player Races')
plt.show()

#%%
weapon = plt.barh(p_weapons['Weapons'][0:10],p_weapons['Counts'][0:10])
plt.bar_label(weapon, fmt='%.0f')
plt.title('Top 10 Player Weapons')
plt.xlabel('Counts')
plt.ylabel('Weapons')
plt.show()

back = plt.barh(bg['Background'],bg['Counts'])
plt.bar_label(back, fmt='%.0f')
plt.title('Player Backgrounds')
plt.xlabel('Counts')
plt.ylabel('Backgrounds')
plt.show()
