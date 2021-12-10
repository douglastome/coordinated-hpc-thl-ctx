'''
Copyright 2021  Douglas Feitosa Tomé

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

matplotlib.use('Agg')

class Results:
 def __init__(self):
  self.random_trial = [] #average
  self.region = [] #hue
  self.neuron_type = []
  self.learn_time = []
  self.consolidation_time = []
  self.phase = [] #style
  self.pattern = []
  self.metric = []
  self.score = [] #y


 def add_result(self, random_trial, region, neuron_type, learn_time, consolidation_time, phase, pattern, metric, score):
  self.random_trial.append(random_trial)
  self.region.append(region)
  self.neuron_type.append(neuron_type)
  self.learn_time.append(learn_time)
  self.consolidation_time.append(consolidation_time)
  self.phase.append(phase)
  self.pattern.append(pattern)
  self.metric.append(metric)
  self.score.append(score)


 def build_data_frame(self):
  dd = {'random_trial' : self.random_trial,
        'region' : self.region,
        'neuron_type' : self.neuron_type,
        'learn_time' : self.learn_time,
        'consolidation_time' : self.consolidation_time,
        'phase' : self.phase,
        'pattern' : self.pattern,
        'metric' : self.metric,
        'score' : self.score}
  self.df = pd.DataFrame(dd)


 def save_data_frame(self, datadir, filename):
  #filename = filename + '.csv'
  self.df.to_csv(os.path.join(datadir, filename))
  print("Saved results data frame in %s"%filename)


 def load_df(self, datadir, filename):
  self.df = pd.read_csv(os.path.join(datadir, filename))


 def select_rows(self, *args):
  select_rows = np.full((len(self.df)), True)
  '''
  print('select_rows')
  print('args', type(args))  
  print(args)
  print('select', type(select), select.shape)
  print(select)
  '''
  for col,values in args:
   '''print ('col,value', col, value)'''
   select = np.full((len(self.df)), False)
   for value in values:
    select = select | (self.df[col] == value)
   select_rows = select_rows & select
  return self.df[select_rows]

 def plot_line(self, x, y, hue, style, ci, data, datadir, filename, xlabel, ylabel):
  plt.rcParams.update({'font.size': 6})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'Verdana'})

  #fig = plt.figure(dpi=300)
  #fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
  fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
  #palette = sns.color_palette("muted", n_colors=4)
  #palette = sns.color_palette(['#4878d0', '#ee854a', '#6acc64', '#8c613c'])
  palette = sns.color_palette(['#4878d0', '#ee854a', '#6acc64'])
  ax = None
  if style == None:
   ax= sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data, palette=palette)
  else:
   ax = sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data, palette=palette)
   
  ax.get_legend().remove()
  
  #box = ax.get_position()
  #ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
  #ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

  #plt.xlabel(xlabel + ' time (min)')
  plt.xlabel(xlabel + ' time (h)')
  plt.ylabel(ylabel)
  #ax.set_xticks(list(set(data[x])))
  
  #plt.title("")
  plt.ylim((0, 100.05))
  #plt.xlim((-0.05, 12.05))
  sns.despine()
  plt.tight_layout()
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print('Saved results plot', filename)
  plt.close(fig)

 '''
 def plot_line(self, x, y, hue, style, ci, data, datadir, filename, xlabel, ylabel):
  fig = plt.figure()
  plt.rcParams.update({'font.size': 13})
  ax = None
  if style == None:
   ax= sns.lineplot(x=x, y=y, hue=hue, ci=ci, data=data)
  else:
   ax = sns.lineplot(x=x, y=y, hue=hue, style=style, ci=ci, data=data)
   
  #ax.get_legend().remove()
  
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.80, box.height]) # resize position
  ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)

  plt.xlabel(xlabel + ' time (min)')
  plt.ylabel(ylabel)
  #ax.set_xticks(list(set(data[x])))
  
  #plt.title("")
  plt.ylim((-0.05, 1.05))
  sns.despine()
  plt.savefig(os.path.join(datadir, filename))
  print('Saved results plot', filename)
  plt.close(fig)
 '''

 def plot(self, x, y, hue, style, ci, datadir, filename, xlabel, ylabel, *args):
  '''
  print('plot args', type(args))
  print(args)
  '''
  data = self.select_rows(*args)
  #data[x] = data[x] / 60.0
  data[x] = data[x] / 3600.0
  data[y] = data[y] * 100
  '''
  print("\nFiltered data frame for plot %s:"%filename)
  print(data)
  '''
  self.plot_line(x, y, hue, style, ci, data, datadir, filename, xlabel, ylabel)
