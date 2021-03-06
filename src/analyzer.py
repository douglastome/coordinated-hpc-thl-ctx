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

import sys
import os
import math
from sklearn.decomposition import NMF, PCA
from scipy.io import mmread, mmwrite
from scipy.stats import kstest
from itertools import combinations, product

# Import Auryn tools
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *

from helper import *


class ActivityStats:
 def __init__(self, time, area_rate):
  self.time = time
  self.area_rate = area_rate


class StimStats:
 def __init__(self, stim_intervals, stim_spikes, rates_per_stim_presentation):
  self.stim_intervals = stim_intervals
  self.stim_spikes = stim_spikes
  self.rates_per_stim_presentation = rates_per_stim_presentation


class PatternPresentation:
 def __init__(self, pattern):
  self.pattern = pattern
  self.stim = None
  self.t_start = None
  self.t_stop = None
  self.presentation = None
  self.duration = 0

 def update(self, stim, t_start, t_stop, presentation, duration):
  self.stim = stim
  self.t_start = t_start
  self.t_stop = t_stop
  self.presentation = presentation
  self.duration = duration

 def print_presentation(self):
  print("pattern %d stim %d: %f-%f (presentation %d) (duration %f)"%(self.pattern, self.stim, self.t_start, self.t_stop, self.presentation, self.duration))


class PrePost:
 def __init__(self,
              connection,
              pre_area,
              post_area,
              pre_neurons=None,
              post_neurons=None,
              pre_ca_partitions=None,
              post_ca_partitions=None,
              pre_ca=None,
              post_ca=None,
              pre_cluster=None,
              post_cluster=None):
  self.connection = connection
  self.pre_area = pre_area
  self.post_area = post_area
  self.pre_neurons = pre_neurons
  self.post_neurons = post_neurons
  self.pre_ca_partitions = pre_ca_partitions
  self.post_ca_partitions = post_ca_partitions
  self.pre_ca = pre_ca
  self.post_ca = post_ca
  self.pre_cluster = pre_cluster
  self.post_cluster = post_cluster


 def pprint(self):
  print('pre_ca', self.pre_ca)
  print('post_ca', self.post_ca)

 def get_wmat_area(self):
  if self.pre_area == self.post_area:
   return self.pre_area
  else:
   return self.pre_area + '_' + self.post_area

 def get_pre_neuron_type(self):
  return self.connection[0]

 def get_post_neuron_type(self):
  return self.connection[1]

 def get_neuron_cluster(self, neuron, pre_post):
  clusters = getattr(self, pre_post + '_cluster')
  neuron_clusters = ''
  for i,cluster in enumerate(clusters):
   if neuron in cluster:
    if len(neuron_clusters) == 0:
     neuron_clusters = neuron_clusters + str(i)
    else:
     neuron_clusters = neuron_clusters + '-' + str(i)
  return neuron_clusters


class Analyzer:
 def __init__(self,
              trial,
              brain_area,
              other_areas,
              phase,
              nb_exc_neurons,
              nb_inh_neurons,
              nb_neurons_stim_replay,
              nb_neuron_cons_stim_replay,
              has_stim_brain_area,
              has_area_area,
              has_rep,
              rundir,
              trialdir,
              prefix,
              full_prefix,
              has_stim_presentation,
              nb_stim,
              t_start,
              t_stop,
              integration_time_step,
              ids_cell_assemb,
              simtime_assemb,
              stim_pat,
              stimtime,
              stimdata,
              plot_exc_neurons,
              plot_inh_neurons,
              exc_record_rank,
              inh_record_rank,
              exc_ampa_nmda_ratio,
              inh_ampa_nmda_ratio,
              num_mpi_ranks,
              cell_assemb_method,
              file_prefix_assemb,
              file_prefix_hm,
              exc_min_rate,
              inh_min_rate,
              bin_size,
              nb_patterns,
              min_weight,
              has_inh_analysis,
              has_recall_metrics,
              has_weight_metrics,
              has_plots,
              has_metrics_plots,
              has_rates_across_stim_plots,
              has_neuron_vi_plots,
              has_cell_assemb_plots,
              has_spike_stats_plots,
              has_activity_stats,
              has_activity_plots,
              has_spike_raster_stats,
              has_weight_plots,
              has_rf_plots,
              colors,
              zoom_range,
              has_recall_metrics_file,
              has_weight_metrics_file,
              u_rest,
              u_exc,
              u_inh,
              simtime,
              simtime_learn=None,
              simtime_consolidation=None,
              reptime=None,
              repdata=None,
              has_rep_presentation=False,
              stim=''):
  self.trial = trial
  self.brain_area = brain_area
  self.other_areas = other_areas
  self.phase = phase
  self.nb_exc_neurons = nb_exc_neurons
  self.nb_inh_neurons = nb_inh_neurons
  self.has_stim_brain_area = has_stim_brain_area
  self.has_area_area = has_area_area
  self.has_rep = has_rep
  self.rundir = rundir
  self.trialdir = trialdir
  self.prefix = prefix
  self.full_prefix = full_prefix
  self.has_stim_presentation = has_stim_presentation
  self.nb_stim = nb_stim
  self.t_start = t_start
  self.t_stop = t_stop
  self.integration_time_step = integration_time_step
  self.ids_cell_assemb = ids_cell_assemb
  self.simtime_assemb = simtime_assemb
  self.stim_pat = stim_pat
  self.stimtime = stimtime
  self.stimdata = stimdata
  self.plot_exc_neurons = plot_exc_neurons
  self.plot_inh_neurons = plot_inh_neurons
  self.exc_record_rank = exc_record_rank
  self.inh_record_rank = inh_record_rank
  self.exc_ampa_nmda_ratio = exc_ampa_nmda_ratio
  self.inh_ampa_nmda_ratio = inh_ampa_nmda_ratio

  # Shared by all analyzers
  self.nb_neurons_stim_replay = nb_neurons_stim_replay
  self.nb_neuron_cons_stim_replay = nb_neuron_cons_stim_replay
  self.num_mpi_ranks = num_mpi_ranks
  self.cell_assemb_method = cell_assemb_method
  self.file_prefix_assemb = file_prefix_assemb
  self.file_prefix_hm = file_prefix_hm
  self.exc_min_rate = exc_min_rate
  self.inh_min_rate = inh_min_rate
  self.bin_size = bin_size
  self.nb_patterns = nb_patterns
  self.min_weight = min_weight
  self.has_inh_analysis = has_inh_analysis
  self.has_recall_metrics = has_recall_metrics
  self.has_weight_metrics = has_weight_metrics
  self.has_plots = has_plots
  self.has_metrics_plots = has_metrics_plots
  self.has_rates_across_stim_plots = has_rates_across_stim_plots
  self.has_neuron_vi_plots = has_neuron_vi_plots
  self.has_cell_assemb_plots = has_cell_assemb_plots
  self.has_spike_stats_plots = has_spike_stats_plots
  self.has_activity_stats = has_activity_stats
  self.has_activity_plots = has_activity_plots
  self.has_spike_raster_stats = has_spike_raster_stats
  self.has_weight_plots = has_weight_plots
  self.has_rf_plots = has_rf_plots
  self.colors = colors
  self.zoom_range = zoom_range
  self.has_recall_metrics_file = has_recall_metrics_file
  self.has_weight_metrics_file = has_weight_metrics_file
  self.u_rest = u_rest
  self.u_exc = u_exc
  self.u_inh = u_inh
  self.area_color = 'dimgray'
  self.no_stim_color = 'black'
  
  self.simtime = simtime
  self.simtime_learn = simtime_learn
  self.simtime_consolidation = simtime_consolidation
  self.reptime = reptime
  self.repdata = repdata
  self.has_rep_presentation = has_rep_presentation

  '''
  if stim == 'bar_a.pat':
   stimfiles = ['bar_a0.pat', 'bar_a1.pat', 'bar_a2.pat', 'bar_a3.pat']
  elif stim == 'rand_pat_4_a.pat'
  '''
  
  self.stim_pats = [[] for _ in range(self.nb_patterns)]
  #print('self.stim_pats', self.stim_pats)
  patfile = open(os.path.join(self.rundir, 'src', 'data', stim), 'r')
  st = -1
  for line in patfile.readlines():
   if line.startswith('#'):
    st += 1
   if not line.startswith('#') and len(line) > 1:
    neuron, value = line.split(' ')
    neuron = int(neuron)
    self.stim_pats[st].append(neuron)
   #print(self.stim_pats[stim])

  '''
  self.stimfiles = stimfiles[:self.nb_patterns]
  #print('self.stimfiles', self.stimfiles)
  self.stim_pats = [[] for _ in range(self.nb_patterns)]
  #print('self.stim_pats', self.stim_pats)
  for stim,stimfile in enumerate(self.stimfiles):
   #print('stim', stim, 'stimfile', stimfile)
   patfile = open(os.path.join(self.rundir, 'src', 'data', stimfile), 'r')
   for line in patfile.readlines():
    if not line.startswith('#') and len(line) > 1:
     neuron, value = line.split(' ')
     neuron = int(neuron)
     self.stim_pats[stim].append(neuron)
   #print(self.stim_pats[stim])
  '''
  
  _, self.stim_partitions = get_ca_partitions(self.nb_exc_neurons, self.nb_patterns, self.stim_pats)
  self.stim_clusters = [self.stim_partitions[0]] + self.stim_pats


 def get_nb_neurons(self, neuron_type):
  return getattr(self, 'nb_' + neuron_type + '_neurons')


 def get_plot_neurons(self, neuron_type):
  return getattr(self, 'plot_' + neuron_type + '_neurons')


 def get_min_rate(self, neuron_type):
  return getattr(self, neuron_type + '_min_rate')


 def get_record_rank(self, neuron_type):
  return getattr(self, neuron_type + '_record_rank')


 def get_ampa_nmda_ratio(self, neuron_type):
  return getattr(self, neuron_type + '_ampa_nmda_ratio')


 def get_mem_name(self, neuron_type):
  return neuron_type + '_mem'

 
 def get_mem(self, neuron_type):
  return getattr(self, self.get_mem_name(neuron_type))


 def set_mem(self, neuron_type, mem):
  setattr(self, self.get_mem_name(neuron_type), mem)


 def has_mem(self, neuron_type):
  return hasattr(self, self.get_mem_name(neuron_type))


 def get_thr_name(self, neuron_type):
  return neuron_type + '_thr'

 
 def get_thr(self, neuron_type):
  return getattr(self, self.get_thr_name(neuron_type))


 def set_thr(self, neuron_type, thr):
  setattr(self, self.get_thr_name(neuron_type), thr)


 def has_thr(self, neuron_type):
  return hasattr(self, self.get_thr_name(neuron_type))


 def get_gampa_name(self, neuron_type):
  return neuron_type + '_gampa'

 
 def get_gampa(self, neuron_type):
  return getattr(self, self.get_gampa_name(neuron_type))


 def set_gampa(self, neuron_type, gampa):
  setattr(self, self.get_gampa_name(neuron_type), gampa)


 def has_gampa(self, neuron_type):
  return hasattr(self, self.get_gampa_name(neuron_type))


 def get_gnmda_name(self, neuron_type):
  return neuron_type + '_gnmda'

 
 def get_gnmda(self, neuron_type):
  return getattr(self, self.get_gnmda_name(neuron_type))


 def set_gnmda(self, neuron_type, gnmda):
  setattr(self, self.get_gnmda_name(neuron_type), gnmda)


 def has_gnmda(self, neuron_type):
  return hasattr(self, self.get_gnmda_name(neuron_type))


 def get_ggaba_name(self, neuron_type):
  return neuron_type + '_ggaba'

 
 def get_ggaba(self, neuron_type):
  return getattr(self, self.get_ggaba_name(neuron_type))


 def set_ggaba(self, neuron_type, ggaba):
  setattr(self, self.get_ggaba_name(neuron_type), ggaba)


 def has_ggaba(self, neuron_type):
  return hasattr(self, self.get_ggaba_name(neuron_type))


 def get_gadapt1_name(self, neuron_type):
  return neuron_type + '_gadapt1'

 
 def get_gadapt1(self, neuron_type):
  return getattr(self, self.get_gadapt1_name(neuron_type))


 def set_gadapt1(self, neuron_type, gadapt1):
  setattr(self, self.get_gadapt1_name(neuron_type), gadapt1)


 def has_gadapt1(self, neuron_type):
  return hasattr(self, self.get_gadapt1_name(neuron_type))


 def get_gadapt2_name(self, neuron_type):
  return neuron_type + '_gadapt2'

 
 def get_gadapt2(self, neuron_type):
  return getattr(self, self.get_gadapt2_name(neuron_type))


 def set_gadapt2(self, neuron_type, gadapt2):
  setattr(self, self.get_gadapt2_name(neuron_type), gadapt2)


 def has_gadapt2(self, neuron_type):
  return hasattr(self, self.get_gadapt2_name(neuron_type))


 def get_gexc_name(self, neuron_type):
  return neuron_type + '_gexc'

 
 def get_gexc(self, neuron_type):
  return getattr(self, self.get_gexc_name(neuron_type))


 def set_gexc(self, neuron_type, gexc):
  setattr(self, self.get_gexc_name(neuron_type), gexc)


 def has_gexc(self, neuron_type):
  return hasattr(self, self.get_gexc_name(neuron_type))


 def get_ginh_name(self, neuron_type):
  return neuron_type + '_ginh'

 
 def get_ginh(self, neuron_type):
  return getattr(self, self.get_ginh_name(neuron_type))


 def set_ginh(self, neuron_type, ginh):
  setattr(self, self.get_ginh_name(neuron_type), ginh)


 def has_ginh(self, neuron_type):
  return hasattr(self, self.get_ginh_name(neuron_type))


 def get_i_exc_name(self, neuron_type):
  return neuron_type + '_i_exc'

 
 def get_i_exc(self, neuron_type):
  return getattr(self, self.get_i_exc_name(neuron_type))


 def set_i_exc(self, neuron_type, i_exc):
  setattr(self, self.get_i_exc_name(neuron_type), i_exc)


 def has_i_exc(self, neuron_type):
  return hasattr(self, self.get_i_exc_name(neuron_type))


 def get_i_leak_name(self, neuron_type):
  return neuron_type + '_i_leak'

 
 def get_i_leak(self, neuron_type):
  return getattr(self, self.get_i_leak_name(neuron_type))


 def set_i_leak(self, neuron_type, i_leak):
  setattr(self, self.get_i_leak_name(neuron_type), i_leak)


 def has_i_leak(self, neuron_type):
  return hasattr(self, self.get_i_leak_name(neuron_type))


 def get_i_inh_name(self, neuron_type):
  return neuron_type + '_i_inh'

 
 def get_i_inh(self, neuron_type):
  return getattr(self, self.get_i_inh_name(neuron_type))


 def set_i_inh(self, neuron_type, i_inh):
  setattr(self, self.get_i_inh_name(neuron_type), i_inh)


 def has_i_inh(self, neuron_type):
  return hasattr(self, self.get_i_inh_name(neuron_type))


 def get_sfo_name(self, neuron_type):
  return neuron_type + '_sfo'


 def has_sfo(self, neuron_type):
  return hasattr(self, self.get_sfo_name(neuron_type))


 def get_sfo(self, neuron_type):
  return getattr(self, self.get_sfo_name(neuron_type))


 def set_sfo(self, neuron_type, sfo):
  setattr(self, self.get_sfo_name(neuron_type), sfo)


 def get_binned_spike_counts_name(self, neuron_type):
  return neuron_type + '_binned_spike_counts'


 def has_binned_spike_counts(self, neuron_type):
  return hasattr(self, self.get_binned_spike_counts_name(neuron_type))


 def get_binned_spike_counts(self, neuron_type):
  return getattr(self, self.get_binned_spike_counts_name(neuron_type))


 def set_binned_spike_counts(self, neuron_type, binned_spike_counts):
  setattr(self, self.get_binned_spike_counts_name(neuron_type), binned_spike_counts)


 def get_stim_spikes_name(self, neuron_type):
  return neuron_type + '_stim_spikes'


 def get_stim_spikes(self, neuron_type):
  return getattr(self, self.get_stim_spikes_name(neuron_type))


 def set_stim_spikes(self, neuron_type, stim_spikes):
  setattr(self, self.get_stim_spikes_name(neuron_type), stim_spikes)


 def get_rates_per_stim_name(self, neuron_type):
  return neuron_type + '_rates_per_stim_presentation'


 def has_rates_per_stim(self, neuron_type):
  return hasattr(self, self.get_rates_per_stim_name(neuron_type))


 def get_rates_per_stim(self, neuron_type):
  return getattr(self, self.get_rates_per_stim_name(neuron_type))


 def set_rates_per_stim(self, neuron_type, rates_per_stim):
  setattr(self, self.get_rates_per_stim_name(neuron_type), rates_per_stim)


 def get_rates_across_stim_name(self, neuron_type):
  return neuron_type + '_rates_across_stim_presentations'


 def has_rates_across_stim(self, neuron_type):
  return hasattr(self, self.get_rates_across_stim_name(neuron_type))


 def get_rates_across_stim(self, neuron_type):
  return getattr(self, self.get_rates_across_stim_name(neuron_type))


 def set_rates_across_stim(self, neuron_type, rates_across_stim):
  setattr(self, self.get_rates_across_stim_name(neuron_type), rates_across_stim)


 def get_cell_assemblies_name(self, neuron_type):
  return neuron_type + '_cell_assemblies'


 def get_cas(self, neuron_type):
  return getattr(self, self.get_cell_assemblies_name(neuron_type))


 def set_cas(self, neuron_type, cas):
  setattr(self, self.get_cell_assemblies_name(neuron_type), cas)


 def get_ca_partitions_name(self, neuron_type):
  return neuron_type + '_ca_partitions'


 def get_ca_partitions(self, neuron_type):
  return getattr(self, self.get_ca_partitions_name(neuron_type))


 def set_ca_partitions(self, neuron_type, cap):
  setattr(self, self.get_ca_partitions_name(neuron_type), cap)


 def get_stim_combinations_name(self, neuron_type):
  return neuron_type + '_stim_combinations'


 def get_stim_combinations(self, neuron_type):
  return getattr(self, self.get_stim_combinations_name(neuron_type))


 def set_stim_combinations(self, neuron_type, stim_comb):
  setattr(self, self.get_stim_combinations_name(neuron_type), stim_comb)


 def get_clusters_name(self, neuron_type):
  return neuron_type + '_clusters'


 def get_clusters(self, neuron_type):
  return getattr(self, self.get_clusters_name(neuron_type))


 def set_clusters(self, neuron_type, clusters):
  setattr(self, self.get_clusters_name(neuron_type), clusters)


 def get_area_rate_name(self, neuron_type):
  return neuron_type + '_area_rate'


 def get_area_rate(self, neuron_type):
  return getattr(self, self.get_area_rate_name(neuron_type))


 def set_area_rate(self, neuron_type, area_rate):
  setattr(self, self.get_area_rate_name(neuron_type), area_rate)


 def get_ca_rates_name(self, neuron_type):
  return neuron_type + '_ca_rates'


 def get_ca_rates(self, neuron_type):
  return getattr(self, self.get_ca_rates_name(neuron_type))


 def set_ca_rates(self, neuron_type, ca_rates):
  setattr(self, self.get_ca_rates_name(neuron_type), ca_rates)


 def get_cap_rates_name(self, neuron_type):
  return neuron_type + '_cap_rates'


 def get_cap_rates(self, neuron_type):
  return getattr(self, self.get_cap_rates_name(neuron_type))


 def set_cap_rates(self, neuron_type, cap_rates):
  setattr(self, self.get_cap_rates_name(neuron_type), cap_rates)


 def get_spike_raster_name(self, neuron_type):
  return neuron_type + '_spike_raster'


 def has_spike_raster(self, neuron_type):
  return hasattr(self, self.get_spike_raster_name(neuron_type))


 def get_spike_raster(self, neuron_type):
  return getattr(self, self.get_spike_raster_name(neuron_type))


 def set_spike_raster(self, neuron_type, spike_raster):
  setattr(self, self.get_spike_raster_name(neuron_type), spike_raster)


 def get_spike_train_name(self, neuron_type):
  return neuron_type + '_spike_train'


 def has_spike_train(self, neuron_type):
  return hasattr(self, self.get_spike_train_name(neuron_type))


 def get_spike_train(self, neuron_type):
  return getattr(self, self.get_spike_train_name(neuron_type))


 def set_spike_train(self, neuron_type, spike_train):
  setattr(self, self.get_spike_train_name(neuron_type), spike_train)


 def get_stim_filename(self, neuron_type):
  return getattr(self, neuron_type + '_stim_filename')


 def has_stim_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_stim_file')


 def get_rates_across_stim_filename(self, neuron_type):
  return getattr(self, neuron_type + '_rates_across_stim_filename')


 def has_rates_across_stim_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_rates_across_stim_file')


 def get_cell_assembly_filename(self, neuron_type):
  return getattr(self, neuron_type + '_ca_filename')


 def has_cell_assembly_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_cell_assembly_file')


 def get_activity_filename(self, neuron_type):
  return getattr(self, neuron_type + '_activity_filename')


 def has_activity_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_activity_file')


 def get_ca_rates_filename(self, neuron_type):
  return getattr(self, neuron_type + '_ca_rates_filename')


 def has_ca_rates_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_ca_rates_file')


 def get_cap_rates_filename(self, neuron_type):
  return getattr(self, neuron_type + '_cap_rates_filename')


 def has_cap_rates_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_cap_rates_file')


 def get_spike_raster_filename(self, neuron_type):
  return getattr(self, neuron_type + '_spike_raster_filename')


 def has_spike_raster_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_spike_raster_file')


 def get_spike_train_filename(self, neuron_type):
  return getattr(self, neuron_type + '_spike_train_filename')


 def has_spike_train_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_spike_train_file')


 def get_binned_spike_counts_filename(self, neuron_type):
  return getattr(self, neuron_type + '_binned_spike_counts_filename')


 def has_binned_spike_counts_file(self, neuron_type):
  return getattr(self, 'has_' + neuron_type + '_binned_spike_counts_file')


 def has_stim(self):
  self.exc_stim_filename = 'exc.stim.' + self.full_prefix + '.' + self.brain_area + '.pkl'
  self.has_exc_stim_file = os.path.isfile(os.path.join(self.trialdir, self.exc_stim_filename))
  if self.has_inh_analysis:
   self.inh_stim_filename = 'inh.stim.' + self.full_prefix + '.' + self.brain_area + '.pkl'
   self.has_inh_stim_file = os.path.isfile(os.path.join(self.trialdir, self.inh_stim_filename))

   
 def has_rates_across_stim_presentations(self):
  self.exc_rates_across_stim_filename = "exc-neuron-rates-across-stim-presentations-" + self.full_prefix + '-' + self.brain_area + '.pkl'
  self.has_exc_rates_across_stim_file = os.path.isfile(os.path.join(self.trialdir, self.exc_rates_across_stim_filename))
  if self.has_inh_analysis:
   self.inh_rates_across_stim_filename = "inh-neuron-rates-across-stim-presentations-" + self.full_prefix + '-' + self.brain_area + '.pkl'
   self.has_inh_rates_across_stim_file = os.path.isfile(os.path.join(self.trialdir, self.inh_rates_across_stim_filename))


 def has_cell_assemblies(self):
  self.exc_ca_filename = self.file_prefix_assemb + '-' + str(self.simtime_assemb) + '.' + self.cell_assemb_method + ".pat.exc." + self.brain_area
  self.has_exc_cell_assembly_file = os.path.isfile(os.path.join(self.trialdir, self.exc_ca_filename))
  if self.has_inh_analysis:
   self.inh_ca_filename = self.file_prefix_assemb + '-' + str(self.simtime_assemb) + '.' + self.cell_assemb_method + ".pat.inh." + self.brain_area
   self.has_inh_cell_assembly_file = os.path.isfile(os.path.join(self.trialdir, self.inh_ca_filename))

 def has_replay(self):
  #self.rep_filename = 'rep.' + self.full_prefix + '.' + self.brain_area + '.pkl'
  #self.has_rep_file = os.path.isfile(os.path.join(self.trialdir, self.rep_filename))
  
  #self.replay_pat_filename = self.file_prefix_assemb + '-' + str(self.simtime_assemb) + '.' + self.cell_assemb_method + ".replay.pat." + self.brain_area
  #self.replay_pat_filename = "replay.pat." + self.brain_area

  self.replay_pat_filename = self.file_prefix_assemb + '-' + str(self.simtime_assemb) + '.' + self.cell_assemb_method + ".replay.pat"
  
  self.replay_wmat_filename = self.file_prefix_assemb + '-' + str(self.simtime_assemb) + '.' + self.cell_assemb_method + ".replay.wmat." + self.brain_area
  
  self.has_rep_pat_file = os.path.isfile(os.path.join(self.trialdir, self.replay_pat_filename))
  self.has_rep_wmat_file = os.path.isfile(os.path.join(self.trialdir, self.replay_wmat_filename + '.mtx'))


 def has_activity(self):
  self.exc_activity_filename = 'exc.activity.' + self.full_prefix + '.' + '.' + self.brain_area + '.pkl'
  self.has_exc_activity_file = os.path.isfile(os.path.join(self.trialdir, self.exc_activity_filename))
  if self.has_inh_analysis:
   self.inh_activity_filename = 'inh.activity.' + self.full_prefix + '.' + '.' + self.brain_area + '.pkl'
   self.has_inh_activity_file = os.path.isfile(os.path.join(self.trialdir, self.inh_activity_filename))


 def has_ca_rates(self):
  self.exc_ca_rates_filename = 'exc_ca_rates.' + self.full_prefix + '.' + self.cell_assemb_method + '-' + str(self.simtime_assemb) + '.' + self.brain_area + '.pkl'
  self.has_exc_ca_rates_file = os.path.isfile(os.path.join(self.trialdir, self.exc_ca_rates_filename))
  if self.has_inh_analysis:
   self.inh_ca_rates_filename = 'inh_ca_rates.' + self.full_prefix + '.' + self.cell_assemb_method + '-' + str(self.simtime_assemb) + '.' + self.brain_area + '.pkl'
   self.has_inh_ca_rates_file = os.path.isfile(os.path.join(self.trialdir, self.inh_ca_rates_filename))


 def has_cap_rates(self):
  self.exc_cap_rates_filename = 'exc_cap_rates.' + self.full_prefix + '.' + self.cell_assemb_method + '-' + str(self.simtime_assemb) + '.' + self.brain_area + '.pkl'
  self.has_exc_cap_rates_file = os.path.isfile(os.path.join(self.trialdir, self.exc_cap_rates_filename))
  if self.has_inh_analysis:
   self.inh_cap_rates_filename = 'inh_cap_rates.' + self.full_prefix + '.' + self.cell_assemb_method + '-' + str(self.simtime_assemb) + '.' + self.brain_area + '.pkl'
   self.has_inh_cap_rates_file = os.path.isfile(os.path.join(self.trialdir, self.inh_cap_rates_filename))


 def has_spike_trains(self):
  self.exc_spike_train_filename = 'exc.spike_train.' + self.full_prefix + '.' + self.brain_area + '.pkl'
  self.has_exc_spike_train_file = os.path.isfile(os.path.join(self.trialdir, self.exc_spike_train_filename))
  if self.has_inh_analysis:
   self.inh_spike_train_filename = 'inh.spike_train.' + self.full_prefix + '.' + self.brain_area + '.pkl'
   self.has_inh_spike_train_file = os.path.isfile(os.path.join(self.trialdir, self.inh_spike_train_filename))


 def has_spike_rasters(self):
  self.exc_spike_raster_filename = 'exc.spike_raster.' + self.full_prefix + '.' + '.' + self.brain_area + '.pkl'
  self.has_exc_spike_raster_file = os.path.isfile(os.path.join(self.trialdir, self.exc_spike_raster_filename))
  if self.has_inh_analysis:
   self.inh_spike_raster_filename = 'inh.spike_raster.' + self.full_prefix + '.' + '.' + self.brain_area + '.pkl'
   self.has_inh_spike_raster_file = os.path.isfile(os.path.join(self.trialdir, self.inh_spike_raster_filename))


 def has_binned_spike_counts_files(self):
  self.exc_binned_spike_counts_filename = 'exc.binned_spike_counts.' + self.full_prefix + '.' + self.brain_area + '.pkl'
  self.has_exc_binned_spike_counts_file = os.path.isfile(os.path.join(self.trialdir, self.exc_binned_spike_counts_filename))
  if self.has_inh_analysis:
   self.inh_binned_spike_counts_filename = 'inh.binned_spike_counts.' + self.full_prefix + '.' + self.brain_area + '.pkl'
   self.has_inh_binned_spike_counts_file = os.path.isfile(os.path.join(self.trialdir, self.inh_binned_spike_counts_filename))


 def has_dynamics_files(self):
  self.has_stim()
  self.has_rates_across_stim_presentations()
  self.has_cell_assemblies()
  if self.has_rep:
   self.has_replay()
  self.has_activity()
  self.has_ca_rates()
  self.has_cap_rates()
  self.has_spike_trains()
  self.has_spike_rasters()
  self.has_binned_spike_counts_files()


 def load_neuron_data(self, neuron_type):
  print("Reading recorded %s %s neuronal data"%(self.brain_area, neuron_type))
  
  if neuron_type == 'exc':
   ntype = 'e'
  elif neuron_type == 'inh':
   ntype = 'i'
  record_rank = self.get_record_rank(neuron_type)
  all_rows = int(self.simtime / self.integration_time_step)
  range_rows = int((self.t_stop - self.t_start) / self.integration_time_step)
  skip_rows = all_rows - range_rows

  mem = np.loadtxt("%s/%s.%d.%s.%s.mem"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_mem(neuron_type, mem)
  
  thr = np.loadtxt("%s/%s.%d.%s.%s.thr"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_thr(neuron_type, thr[:,1])
  
  gampa = np.loadtxt("%s/%s.%d.%s.%s.gampa"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_gampa(neuron_type, gampa[:,1])
  
  gnmda = np.loadtxt("%s/%s.%d.%s.%s.gnmda"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_gnmda(neuron_type, gnmda[:,1])
  
  ggaba = np.loadtxt("%s/%s.%d.%s.%s.ggaba"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_ggaba(neuron_type, ggaba[:,1])
  
  gadapt1 = np.loadtxt("%s/%s.%d.%s.%s.gadapt1"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_gadapt1(neuron_type, gadapt1[:,1])
  
  gadapt2 = np.loadtxt("%s/%s.%d.%s.%s.gadapt2"%(self.trialdir,self.prefix,record_rank,ntype,self.brain_area),skiprows=skip_rows)
  self.set_gadapt2(neuron_type, gadapt2[:,1])
  
  print("Finished reading recorded %s %s neuronal data"%(self.brain_area, neuron_type))
  

 def get_gs(self, neuron_type):
  ampa_nmda_ratio = self.get_ampa_nmda_ratio(neuron_type)
  gampa = self.get_gampa(neuron_type)
  gnmda = self.get_gnmda(neuron_type)
  ggaba = self.get_ggaba(neuron_type)
  gadapt1 = self.get_gadapt1(neuron_type)
  gadapt2 = self.get_gadapt2(neuron_type)
  
  gexc = (ampa_nmda_ratio * gampa) + (1 - ampa_nmda_ratio) * gnmda
  self.set_gexc(neuron_type, gexc)
  
  ginh = ggaba + gadapt1 + gadapt2
  self.set_ginh(neuron_type, ginh)


 def get_vi(self, neuron_type):
  gexc = self.get_gexc(neuron_type)
  ginh = self.get_ginh(neuron_type)
  mem = self.get_mem(neuron_type)
  
  i_exc = gexc * (self.u_exc - mem[:,1])
  self.set_i_exc(neuron_type, i_exc)
  
  i_leak = self.u_rest - mem[:,1]
  self.set_i_leak(neuron_type, i_leak)
  
  i_inh = ginh * (self.u_inh - mem[:,1]) + i_leak
  self.set_i_inh(neuron_type, i_inh)


 def get_neuron_vi(self, neuron_type):
  self.load_neuron_data(neuron_type)
  self.get_gs(neuron_type)
  self.get_vi(neuron_type)


 def plot_vi(self, neuron_type, t_start, t_stop, filename):
  print ("Plotting %s %s neuronal voltage and current..."%(self.brain_area, neuron_type))

  mem = self.get_mem(neuron_type)
  thr = self.get_thr(neuron_type)
  i_exc = self.get_i_exc(neuron_type)
  i_inh = self.get_i_inh(neuron_type)
  
  gs = GridSpec(2,1,height_ratios=[20,14])
  fig = plt.figure(dpi=300)
  ax = plt.subplot(gs[0])
  plt.plot(mem[:,0]*1000, mem[:,1]*1000, color="black", label="V")
  plt.plot(mem[:,0]*1000, thr*1000, color="gray", label="Thr")
  plt.ylabel("mV")
  ax.get_xaxis().set_visible(False)
  plt.xlim((t_start*1000,t_stop*1000))
  plt.legend()
  ax = plt.subplot(gs[1])
  plt.plot(mem[:,0]*1000, i_exc, color="red", label="Exc")
  plt.plot(mem[:,0]*1000, i_inh, color="blue", label="Inh")
  plt.plot(mem[:,0]*1000, i_exc + i_inh, color="green", label="Net")
  ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.75)
  plt.ylabel('I')
  plt.xlabel('Time (ms)')
  plt.xlim((t_start*1000,t_stop*1000))
  plt.legend()
  
  plt.savefig(os.path.join(self.trialdir, filename), format='svg', dpi=300)
  plt.close(fig)
  print ("Saved plot of %s %s neuronal voltage and current"%(self.brain_area, neuron_type))

  
 def plot_vi_traces(self, neuron_type):
  plot_times = [(self.t_start, self.t_stop), (self.t_start, self.t_start + 1), ((self.t_start + self.t_stop)//2, (self.t_start + self.t_stop)//2 + 1), (self.t_stop - 1, self.t_stop)]
  plot_filenames = []
  has_plots = []
  for t_start,t_stop in plot_times:
   filename = self.brain_area + '-' + neuron_type + '-vi-' + self.prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + '.svg'
   has_plot = os.path.isfile(os.path.join(self.trialdir, filename))
   plot_filenames.append(filename)
   has_plots.append(int(has_plot))

  if sum(has_plots) < len(has_plots):
   self.get_neuron_vi(neuron_type)
   for times,filename in zip(plot_times,plot_filenames):
    t_start,t_stop = times[0],times[1]
    self.plot_vi(neuron_type, t_start, t_stop, filename)


 def load_spike_view(self, neuron_type):
  print ("Opening %s %s spike output files..."%(self.brain_area, neuron_type))
  if neuron_type == 'exc':
   neuron = 'e'
  elif neuron_type == 'inh':
   neuron = 'i'
  spkfiles  = ["%s/%s.%i.%s.%s.spk"%(self.trialdir,self.prefix,i,neuron,self.brain_area) for i in range(self.num_mpi_ranks)]
  #print("Spike files:", spkfiles)
  self.set_sfo(neuron_type, AurynBinarySpikeView(spkfiles))


 def delete_spike_view(self):
  if hasattr(self, 'exc_sfo'):
   delattr(self, 'exc_sfo')
  if hasattr(self, 'inh_sfo'):
   delattr(self, 'inh_sfo')


 def load_stim_stats(self, neuron_type):
  print("Loading %s %s stim stats..."%(self.brain_area, neuron_type))
  stim_filename = self.get_stim_filename(neuron_type)
  stim_stats = load_pickled_object(os.path.join(self.trialdir, stim_filename))
  self.stim_intervals = stim_stats.stim_intervals
  self.set_stim_spikes(neuron_type, stim_stats.stim_spikes)
  self.set_rates_per_stim(neuron_type, stim_stats.rates_per_stim_presentation)


 def extract_stim_intervals(self):
  print ("Extracting stimulus presentation intervals...")
  self.stim_intervals = get_stim_intervals(self.nb_stim, self.stimtime, self.stimdata, self.t_start, self.t_stop)


 def extract_stim_evoked_spikes(self, neuron_type):
  print ("Extracting %s %s stimulus-evoked spikes..."%(self.brain_area, neuron_type))
  if not self.has_sfo(neuron_type):
   self.load_spike_view(neuron_type)
  nb_neurons = self.get_nb_neurons(neuron_type)
  sfo = self.get_sfo(neuron_type)
  stim_spikes = get_stim_spikes(self.nb_stim, nb_neurons, self.stim_intervals, sfo)
  self.set_stim_spikes(neuron_type, stim_spikes)


 def compute_rates_per_stim_presentation(self, neuron_type):
  print ("Computing %s %s stimulus-evoked neuronal firing rates..."%(self.brain_area, neuron_type))
  stim_spikes = self.get_stim_spikes(neuron_type)
  rates_per_stim = get_rates_per_stim_presentation(self.nb_stim, self.stim_intervals, stim_spikes)
  self.set_rates_per_stim(neuron_type, rates_per_stim)


 def save_stim_stats(self, neuron_type):
  stim_spikes = self.get_stim_spikes(neuron_type)
  rates_per_stim = self.get_rates_per_stim(neuron_type)
  stim_stats = StimStats(self.stim_intervals, stim_spikes, rates_per_stim)
  stim_filename = self.get_stim_filename(neuron_type)
  save_pickled_object(stim_stats, os.path.join(self.trialdir, stim_filename))
  print("Saved %s %s stim stats"%(self.brain_area, neuron_type))
  
   
 def compute_stim_stats(self, neuron_type):
  self.extract_stim_intervals()
  self.extract_stim_evoked_spikes(neuron_type)
  self.compute_rates_per_stim_presentation(neuron_type)
  self.save_stim_stats(neuron_type)


 def get_stim_stats(self, neuron_type):
  if self.has_stim_file(neuron_type):
   self.load_stim_stats(neuron_type)
  else:
   self.compute_stim_stats(neuron_type)


 def load_rates_across_stim_presentations(self, neuron_type):
  print("Loading %s %s rates across stimulus presentations..."%(self.brain_area, neuron_type))
  rates_across_stim_filename = self.get_rates_across_stim_filename(neuron_type)
  rates_across_stim = load_pickled_object(os.path.join(self.trialdir, rates_across_stim_filename))
  self.set_rates_across_stim(neuron_type, rates_across_stim)


 def compute_rates_across_stim_presentations(self, neuron_type):
  self.get_stim_stats(neuron_type)
  stim_spikes = self.get_stim_spikes(neuron_type)
  rates_across_stim = get_rates_across_stim_presentations(self.nb_stim, self.stim_intervals, stim_spikes)
  self.set_rates_across_stim(neuron_type, rates_across_stim)
  rates_across_stim_filename = self.get_rates_across_stim_filename(neuron_type)
  save_pickled_object(rates_across_stim, os.path.join(self.trialdir, rates_across_stim_filename))
  print("Saved %s %s rates across stimulus presentations"%(self.brain_area, neuron_type))


 def get_rates_across_stim_presentations(self, neuron_type):
  if self.has_rates_across_stim_file(neuron_type):
   self.load_rates_across_stim_presentations(neuron_type)
  else:
   self.compute_rates_across_stim_presentations(neuron_type)


 def plot_rates_hist(self, rates, bins, color, neuron_type, filename):
  plt.rcParams.update({'font.size': 6})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'Verdana'})
 
  print ("Plotting %s %s rates histogram"%(self.brain_area, neuron_type))
  #fig = plt.figure(figsize=(3,1.32), dpi=300)
  #fig = plt.figure(figsize=(1.12,0.84), dpi=300)
  fig = plt.figure(figsize=(1.5,0.84), dpi=300)
  plt.hist(rates, bins=bins, weights=np.ones(len(rates)) / len(rates), color=color)
  plt.axvline(x=10, color='gray', linestyle='--', linewidth=2.00)
  plt.ylim((0,1.05))
  plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
  plt.ylabel('fraction of\n neurons (%)')
  plt.xlabel('firing rate (Hz)')
  #plt.grid(True)
  sns.despine()
  plt.tight_layout()
  plt.savefig(os.path.join(self.trialdir, filename), format='svg', dpi=300)
  plt.close(fig)
  
   
 def plot_rates_across_stim_presentations(self, neuron_type):
  extension = '.svg'
  
  rates_across_stim = self.get_rates_across_stim(neuron_type)
  cell_assemblies = self.get_cas(neuron_type)
  base_filename = self.brain_area + '-' + neuron_type + '-rates-across-stim-presentations-' + self.full_prefix + '-stim-'
  for stim in range(self.nb_stim):
   stim_filename = base_filename + str(stim)
   rates = rates_across_stim[stim]
   max_rate = rates.max()
   max_bin = int(math.ceil(max_rate / 10))
   bins = [i*10 for i in range(max_bin + 1)]
   
   filename = stim_filename + '-area_rates'  + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    self.plot_rates_hist(rates, bins, self.area_color, neuron_type, filename)

   if type(cell_assemblies) != type(None):
    for ca in range(self.nb_patterns):
     filename = stim_filename + '-ca_-' + str(ca) + '_rates'  + extension
     if not os.path.isfile(os.path.join(self.trialdir, filename)):
      self.plot_rates_hist(rates[cell_assemblies[ca]], bins, self.colors[ca], neuron_type, filename)


 def identify_rate_cas(self, neuron_type):
  print('Using firing rate to identify cell assemblies...')
  if not self.has_rates_across_stim(neuron_type):
   self.get_rates_across_stim_presentations(neuron_type)
  rates = self.get_rates_across_stim(neuron_type)
  min_rate = self.get_min_rate(neuron_type)
  cas = get_cell_assembly_rate(self.nb_stim, rates, min_rate)
  self.set_cas(neuron_type, cas)


 def identify_rate_percentile_cas(self, neuron_type, percentile=25):
  print('Using firing rate percentile to identify cell assemblies...')
  if not self.has_rates_per_stim(neuron_type):
   self.get_stim_stats(neuron_type)
  rates_per_stim = self.get_rates_per_stim(neuron_type)
  rates = get_rates_percentile_across_stim_presentations(self.nb_stim, rates_per_stim, percentile)
  min_rate = self.get_min_rate(neuron_type)
  cas = get_cell_assembly_rate(self.nb_stim, rates, min_rate)
  self.set_cas(neuron_type, cas)


 def identify_nmf_cas(self, neuron_type):
  print("Using NMF to identify cell assemblies...")
  if not self.has_sfo(neuron_type):
   self.load_spike_view(neuron_type)
  print("Binning %s %s spike file..."%(self.brain_area, neuron_type))
  sfo = self.get_sfo(neuron_type)
  nb_neurons = self.get_nb_neurons(neuron_type)
  binned_spike_counts = sfo.time_binned_spike_counts(self.t_start, self.t_stop, bin_size=self.bin_size, max_neuron_id=nb_neurons)
  nmf = NMF(n_components=self.nb_patterns)
  nmf.fit(binned_spike_counts)
  cas = get_cell_assembly_nmf(self.nb_patterns, nmf, self.min_weight)
  self.set_cas(neuron_type, cas)


 def save_cell_assemblies(self, neuron_type):
  cas = self.get_cas(neuron_type)
  ca_filename = self.get_cell_assembly_filename(neuron_type)
  save_cell_assembly_file(self.brain_area, cas, self.trialdir, ca_filename)
  print("Saved %s %s cell assemblies"%(self.brain_area, neuron_type))


 def generate_replay(self):
  if not self.has_rep_pat_file:
   generate_replay_pat(self.brain_area, self.exc_cell_assemblies, self.nb_neurons_stim_replay, self.trialdir, self.replay_pat_filename)
  if not self.has_rep_wmat_file:
   generate_replay_wmat(self.brain_area, self.nb_exc_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, self.exc_cell_assemblies, self.trialdir, self.replay_wmat_filename)


 def identify_cell_assemblies(self, neuron_type):
  print ("Identifying %s %s cell assemblies..."%(self.brain_area, neuron_type))
  if self.cell_assemb_method == "rate":
   self.identify_rate_cas(neuron_type)
   #self.identify_rate_percentile_cas(neuron_type)
  elif self.cell_assemb_method == "nmf":
   self.identify_nmf_cas(neuron_type)
  self.save_cell_assemblies(neuron_type)
  if self.has_rep and neuron_type == 'exc':
   self.generate_replay()


 def load_cell_assemblies(self, neuron_type):
  print ("Loading %s %s cell assemblies..."%(self.brain_area, neuron_type))
  ca_filename = self.get_cell_assembly_filename(neuron_type)
  cas = load_cell_assemblies(self.trialdir, ca_filename)
  self.set_cas(neuron_type, cas)


 def join_clusters(self, neuron_type):
  self.set_clusters(neuron_type, [])
  clusters = self.get_clusters(neuron_type)
  ca_partitions = self.get_ca_partitions(neuron_type)
  clusters.append(ca_partitions[0])
  cas = self.get_cas(neuron_type)
  for ca in cas:
   clusters.append(ca)


 def get_stim_comb_cap_clusters(self, neuron_type):
  cell_assemblies = self.get_cas(neuron_type)
  if type(cell_assemblies) != type(None):
   nb_neurons = self.get_nb_neurons(neuron_type)
   stim_combinations, ca_partitions = get_ca_partitions(nb_neurons, self.nb_patterns, cell_assemblies)
   self.set_stim_combinations(neuron_type, stim_combinations)
   self.set_ca_partitions(neuron_type, ca_partitions)
   self.join_clusters(neuron_type)
  else:
   self.set_stim_combinations(neuron_type, None)
   self.set_ca_partitions(neuron_type, None)
   self.set_clusters(neuron_type, None)
  '''
   print("Check cell assembly partitions")
   nb_neurons = 0
   neuron_ca_count_freq = np.zeros(1 + self.nb_patterns)
   print("Stimulus combinations:")
   for sc,cap in zip(self.stim_combinations,self.ca_partitions):
    print(sc)
    nb_neurons += len(cap)
    neuron_ca_count_freq[len(sc)] += len(cap)
   print("Number of neurons in partitions: %d"%nb_neurons)
   print("Number of neurons in %s: %d"%(self.brain_area, self.nb_neurons))
   print("neuron_ca_count_freq")
   print(neuron_ca_count_freq)
   '''


 def get_cell_assemblies(self, neuron_type):
  self.set_cas(neuron_type, None)
  if self.has_cell_assembly_file(neuron_type):
   self.load_cell_assemblies(neuron_type)
  elif self.ids_cell_assemb:
   self.identify_cell_assemblies(neuron_type)
  self.get_stim_comb_cap_clusters(neuron_type)
  
  '''
  print ('Check saving/loading cell assemblies')
  loaded = load_cell_assemblies(self.trialdir, self.ca_filename)
  for ca1,ca2 in zip(self.cell_assemblies, loaded):
   for n1,n2 in zip(ca1, ca2):
    if n1 != n2:
     print(n1, n2)
  '''


 def plot_cell_assemblies(self, neuron_type):
  nb_neurons = self.get_nb_neurons(neuron_type)
  
  filename = self.brain_area + '-' + neuron_type + '-' + self.cell_assemb_method + '-cell-assemb-membership-' + self.full_prefix + '.svg'
  if not os.path.isfile(os.path.join(self.trialdir, filename)):
   clusters = self.get_clusters(neuron_type)
   cluster_count = get_lst_seq_count(clusters)
   plot_ca_count(self.nb_stim, self.colors, cluster_count, nb_neurons, self.brain_area, neuron_type, self.trialdir, filename)

  filename = self.brain_area + '-' + neuron_type + '-' + self.cell_assemb_method + '-cell-assemb-partition-membership-' + self.full_prefix + '.svg'
  if not os.path.isfile(os.path.join(self.trialdir, filename)):
   ca_partitions = self.get_ca_partitions(neuron_type)
   ca_partition_count = get_lst_seq_count(ca_partitions)
   stim_combinations = self.get_stim_combinations(neuron_type)
   plot_ca_partition_count(stim_combinations, ca_partition_count, self.brain_area, neuron_type, self.trialdir, filename)

  #neuron_ca_count = get_neuron_ca_count(self.nb_neurons, self.cell_assemblies)
  #neuron_ca_count_freq = get_neuron_ca_count_freq_old(self.nb_stim, neuron_ca_count)
  #print('neuron_ca_count_freq')
  #print(neuron_ca_count_freq)
  filename = self.brain_area + '-' + neuron_type + '-' + self.cell_assemb_method + '-neuron-cell-assemb-freq-' + self.full_prefix + '.svg'
  if not os.path.isfile(os.path.join(self.trialdir, filename)):
   stim_combinations = self.get_stim_combinations(neuron_type)
   ca_partitions = self.get_ca_partitions(neuron_type)
   neuron_ca_count_freq = get_neuron_ca_count_freq(self.nb_patterns, stim_combinations, ca_partitions)
   plot_neuron_ca_count_freq(self.nb_stim, neuron_ca_count_freq, nb_neurons, self.brain_area, neuron_type, self.trialdir, filename)


 def load_spk_train(self, neuron_type):
  print("Loading %s %s spike train..."%(self.brain_area, neuron_type))
  spike_train_filename = self.get_spike_train_filename(neuron_type)
  spike_train = load_pickled_object(os.path.join(self.trialdir, spike_train_filename))
  self.set_spike_train(neuron_type, spike_train)


 def extract_spk_train(self, neuron_type):
  if not self.has_sfo(neuron_type):
   self.load_spike_view(neuron_type)
  print ("Extracting %s %s spike train..."%(self.brain_area, neuron_type))
  nb_neurons = self.get_nb_neurons(neuron_type)
  sfo = self.get_sfo(neuron_type)
  spike_train = sfo.get_spikes(self.t_start, self.t_stop, nb_neurons)
  self.set_spike_train(neuron_type, spike_train)
  spike_train_filename = self.get_spike_train_filename(neuron_type)
  save_pickled_object(spike_train, os.path.join(self.trialdir, spike_train_filename))
  print("Saved %s %s spike train"%(self.brain_area, neuron_type))
   
  
 def get_spk_train(self, neuron_type):
  if self.has_spike_train_file(neuron_type):
   self.load_spk_train(neuron_type)
  else:
   self.extract_spk_train(neuron_type)


 def plot_spontaneous_activity_spike_stats(self, neuron_type):
  extension ='.svg'
  
  if self.has_stim_presentation:
   max_t_start, max_t_stop = get_max_inter_stim_interval(self.stimtime, self.stimdata, self.t_start, self.t_stop)
   filename = self.brain_area + '-' + neuron_type + '-spike_stats-area-' + self.full_prefix + '-t_start_' + str(max_t_start) + '-t_stop_' + str(max_t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    if not self.has_sfo(neuron_type):
     self.load_spike_view(neuron_type)
    sfo = self.get_sfo(neuron_type)
    nb_neurons = self.get_nb_neurons(neuron_type)
    max_spikes = sfo.get_spikes(max_t_start, max_t_stop, nb_neurons)
    plot_spike_stats(max_spikes, self.area_color, self.brain_area, neuron_type, self.trialdir, filename)  
  elif self.has_rep_presentation:
   max_t_start, max_t_stop = get_max_inter_stim_interval(self.reptime, self.repdata, self.t_start, self.t_stop)
   filename = self.brain_area + '-' + neuron_type + '-spike_stats-area-' + self.full_prefix + '-t_start_' + str(max_t_start) + '-t_stop_' + str(max_t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    if not self.has_sfo(neuron_type):
     self.load_spike_view(neuron_type)
    sfo = self.get_sfo(neuron_type)
    nb_neurons = self.get_nb_neurons(neuron_type)
    max_spikes = sfo.get_spikes(max_t_start, max_t_stop, nb_neurons)
    plot_spike_stats(max_spikes, self.area_color, self.brain_area, neuron_type, self.trialdir, filename)
  else:
   filename = self.brain_area + '-' + neuron_type + '-spike_stats-area-' + self.full_prefix + '-t_start_' + str(self.t_start) + '-t_stop_' + str(self.t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    if not self.has_spike_train(neuron_type):
     self.get_spk_train(neuron_type)
    spikes = self.get_spike_train(neuron_type)
    #plot_spike_stats(spikes, self.area_color, self.brain_area, neuron_type, self.trialdir, filename)
    plot_spike_stats(spikes, 'white', self.brain_area, neuron_type, self.trialdir, filename)


 def get_longest_pattern_presentations(self):
  self.pattern_presentations = [PatternPresentation(pat) for pat in range(self.nb_patterns)]
  for stim in range(self.nb_stim):
   pattern = self.stim_pat[stim]
   for i,interval in enumerate(self.stim_intervals[stim]):
    start,stop = interval
    duration = stop - start
    if duration > self.pattern_presentations[pattern].duration:
     self.pattern_presentations[pattern].update(stim, start, stop, i, duration)


 def plot_stim_evoked_ca_rate_hist(self, neuron_type):
  extension = '.svg'
  
  for pat_pres in self.pattern_presentations:
   pattern = pat_pres.pattern
   stim = pat_pres.stim
   t_start = pat_pres.t_start
   t_stop = pat_pres.t_stop
   presentation = pat_pres.presentation
   rates_per_stim_presentation = self.get_rates_per_stim(neuron_type)
   rates = rates_per_stim_presentation[stim][presentation,:]
   max_rate = rates.max()
   max_bin = int(math.ceil(max_rate / 10))
   bins = [i*10 for i in range(max_bin + 1)]
   for ca in range(self.nb_patterns):
    filename = self.brain_area + '-' + neuron_type + '-stim-evoked-ca-rate-hist-' + self.full_prefix + '-t_start_' + str(t_start) + '-t_stop_' + str(t_stop) + '-pat-' + str(pattern) + '-ca-' + str(ca) + extension
    if not os.path.isfile(os.path.join(self.trialdir, filename)):
     cell_assemblies = self.get_cas(neuron_type)
     ca_rates = rates_per_stim_presentation[stim][presentation, cell_assemblies[ca]]
     print ("Plotting stim-evoked rate histogram of %s %s cell assembly %d:"%(self.brain_area, neuron_type, ca))
     pat_pres.print_presentation()
     self.plot_rates_hist(ca_rates, bins, self.colors[ca], neuron_type, filename)


 def plot_stim_evoked_activity_spike_stats(self, neuron_type):
  self.get_stim_stats(neuron_type)
  self.get_longest_pattern_presentations()
  self.plot_stim_evoked_ca_rate_hist(neuron_type)


 def plot_spike_stats(self, neuron_type):
  self.plot_spontaneous_activity_spike_stats(neuron_type)
  if self.has_stim_presentation:
   self.plot_stim_evoked_activity_spike_stats(neuron_type)


 def load_activity_stats(self, neuron_type):
  print("Loading %s %s activity stats..."%(self.brain_area, neuron_type))
  activity_filename = self.get_activity_filename(neuron_type)
  activity_stats = load_pickled_object(os.path.join(self.trialdir, activity_filename))
  self.time = activity_stats.time
  self.set_area_rate(neuron_type, activity_stats.area_rate)


 def load_spike_raster_sample(self, neuron_type):
  print("Loading %s %s spike raster sample..."%(self.brain_area, neuron_type))
  spike_raster_filename = self.get_spike_raster_filename(neuron_type)
  spike_raster = load_pickled_object(os.path.join(self.trialdir, spike_raster_filename))
  self.set_spike_raster(neuron_type, spike_raster)


 def load_ca_rates(self, neuron_type):
  print("Loading %s %s cell assembly rates..."%(self.brain_area, neuron_type))
  ca_rates_filename = self.get_ca_rates_filename(neuron_type)
  ca_rates = load_pickled_object(os.path.join(self.trialdir, ca_rates_filename))
  self.set_ca_rates(neuron_type, ca_rates)
  

 def load_cap_rates(self, neuron_type):
  print("Loading %s %s cell assembly partition rates..."%(self.brain_area, neuron_type))
  cap_rates_filename = self.get_cap_rates_filename(neuron_type)
  cap_rates = load_pickled_object(os.path.join(self.trialdir, cap_rates_filename))
  self.set_cap_rates(neuron_type, cap_rates)

 
 def load_binned_spike_counts(self, neuron_type):
  print("Loading %s %s binned spike counts..."%(self.brain_area, neuron_type))
  binned_spike_counts_filename = self.get_binned_spike_counts_filename(neuron_type)
  binned_spike_counts = load_pickled_object(os.path.join(self.trialdir, binned_spike_counts_filename))
  self.set_binned_spike_counts(neuron_type, binned_spike_counts)


 def compute_binned_spike_counts(self, neuron_type):
  if not self.has_sfo(neuron_type):
   self.load_spike_view(neuron_type)
  print("Binning %s %s spike file..."%(self.brain_area, neuron_type))
  sfo = self.get_sfo(neuron_type)
  nb_neurons = self.get_nb_neurons(neuron_type)
  binned_spike_counts = sfo.time_binned_spike_counts(self.t_start, self.t_stop, bin_size=self.bin_size, max_neuron_id=nb_neurons)
  self.set_binned_spike_counts(neuron_type, binned_spike_counts)
  binned_spike_counts_filename = self.get_binned_spike_counts_filename(neuron_type)
  #save_pickled_object(binned_spike_counts, os.path.join(self.trialdir, binned_spike_counts_filename))
  print("Saved %s %s binned spike counts"%(self.brain_area, neuron_type))


 def retrieve_binned_spike_counts(self, neuron_type):
  if self.has_binned_spike_counts_file(neuron_type):
   self.load_binned_spike_counts(neuron_type)
  else:
   self.compute_binned_spike_counts(neuron_type)


 def compute_area_rate(self, neuron_type):
  if not self.has_binned_spike_counts(neuron_type):
   self.retrieve_binned_spike_counts(neuron_type)
  print("Computing %s %s area rate..."%(self.brain_area, neuron_type))
  binned_spike_counts = self.get_binned_spike_counts(neuron_type)
  nb_neurons = self.get_nb_neurons(neuron_type)
  area_rate = get_ca_rates(binned_spike_counts / self.bin_size, [list(range(nb_neurons))])
  self.set_area_rate(neuron_type, area_rate)


 def extract_spike_raster_sample(self, neuron_type):
  '''
  raster = np.copy(binned_spike_counts[:, self.plot_neurons])
  for neuron in range(raster.shape[1]):
   for tb in range(raster.shape[0]):
    if raster[tb, neuron] > 0:
     raster[tb, neuron] = neuron
  '''

  if not self.has_spike_train(neuron_type):
   self.get_spk_train(neuron_type)
  spike_train = self.get_spike_train(neuron_type)
  raster = []
  neuron_count = 0
  neuron_map = {}
  plot_neurons = self.get_plot_neurons(neuron_type)
  print ("Extracting %s %s spike raster sample..."%(self.brain_area, neuron_type))
  for time,neuron in spike_train:
   if neuron in plot_neurons:
    if neuron not in neuron_map:
     neuron_map[neuron] = neuron_count
     neuron_count += 1
    raster.append((time, neuron_map[neuron]))
  raster = np.array(raster)
  self.set_spike_raster(neuron_type, raster)

  '''
  print('\n\nraster', raster.shape)
  print(raster)
  '''


 def get_spike_raster_sample(self, neuron_type):
  self.extract_spike_raster_sample(neuron_type)
  spike_raster = self.get_spike_raster(neuron_type)
  spike_raster_filename = self.get_spike_raster_filename(neuron_type)
  save_pickled_object(spike_raster, os.path.join(self.trialdir, spike_raster_filename))
  print("Saved %s %s spike raster sample"%(self.brain_area, neuron_type))


 def get_activity_stats_time(self, rate):
  self.time = np.linspace(self.t_start, self.t_stop, rate[0].shape[0])


 def compute_activity_stats(self, neuron_type):
  self.compute_area_rate(neuron_type)
  area_rate = self.get_area_rate(neuron_type)
  self.get_activity_stats_time(area_rate)
  activity_stats = ActivityStats(self.time, area_rate)
  activity_filename = self.get_activity_filename(neuron_type)
  save_pickled_object(activity_stats, os.path.join(self.trialdir, activity_filename))
  print("Saved %s %s activity stats"%(self.brain_area, neuron_type))


 def compute_ca_rates(self, neuron_type):
  cell_assemblies = self.get_cas(neuron_type)
  if type(cell_assemblies) != type(None):
   if not self.has_binned_spike_counts(neuron_type):
    self.retrieve_binned_spike_counts(neuron_type)
   print("Computing %s %s cell assembly rates..."%(self.brain_area, neuron_type))
   binned_spike_counts = self.get_binned_spike_counts(neuron_type)
   ca_rates = get_ca_rates(binned_spike_counts / self.bin_size, cell_assemblies)
   self.set_ca_rates(neuron_type, ca_rates)
   ca_rates_filename = self.get_ca_rates_filename(neuron_type)
   save_pickled_object(ca_rates, os.path.join(self.trialdir, ca_rates_filename))
  else:
   self.set_ca_rates(neuron_type, None)
  '''
  print ('Check cell assembly rate computation')
  ca_rates_old = get_ca_rates_old(self.binned_spike_counts, self.cell_assemblies, self.bin_size)
  for car1, car2 in zip(self.ca_rates, ca_rates_old):
   print ('check', np.sum(car1 - car2))
  '''

  
 def compute_cap_rates(self, neuron_type):
  ca_partitions = self.get_ca_partitions(neuron_type)
  if type(ca_partitions) != type(None):
   if not self.has_binned_spike_counts(neuron_type):
    self.retrieve_binned_spike_counts(neuron_type)
   print("Computing %s %s cell assembly partition rates..."%(self.brain_area, neuron_type))
   binned_spike_counts = self.get_binned_spike_counts(neuron_type)
   cap_rates = get_ca_rates(binned_spike_counts / self.bin_size, ca_partitions)
   self.set_cap_rates(neuron_type, cap_rates)
   cap_rates_filename = self.get_cap_rates_filename(neuron_type)
   save_pickled_object(cap_rates, os.path.join(self.trialdir, cap_rates_filename))
  else:
   self.set_cap_rates(neuron_type, None)


 def get_area_stats(self, neuron_type):
  if self.has_activity_file(neuron_type):
   self.load_activity_stats(neuron_type)
  elif self.has_activity_stats:
   self.compute_activity_stats(neuron_type)


 def get_cell_assembly_stats(self, neuron_type):
  if self.has_ca_rates_file(neuron_type):
   self.load_ca_rates(neuron_type)
  elif self.has_activity_stats:
   self.compute_ca_rates(neuron_type)


 def get_cell_assembly_partition_stats(self, neuron_type):
  if self.has_cap_rates_file(neuron_type):
   self.load_cap_rates(neuron_type)
  elif self.has_activity_stats:
   self.compute_cap_rates(neuron_type)


 def get_spike_raster_stats(self, neuron_type):
  if self.has_spike_raster_file(neuron_type):
   self.load_spike_raster_sample(neuron_type)
  elif self.has_spike_raster_stats:
   self.get_spike_raster_sample(neuron_type)


 def get_activity_stats(self, neuron_type):
  self.get_area_stats(neuron_type)
  self.get_cell_assembly_stats(neuron_type)
  self.get_cell_assembly_partition_stats(neuron_type)
  self.get_spike_raster_stats(neuron_type)


 def plot_stim_spike_activity(self, neuron_type, t_start, t_stop, stim_type):
  min_rate = self.get_min_rate(neuron_type)
  area_rate = self.get_area_rate(neuron_type)
  ca_rates = self.get_ca_rates(neuron_type)
  cap_rates = self.get_cap_rates(neuron_type)
  if stim_type == 'stim':
   stimtime = self.stimtime
   stimdata = self.stimdata
  elif stim_type == 'rep':
   stimtime= self.reptime
   stimdata = self.repdata

  extension = '.svg'
   
  filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-activity-area-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
  if not os.path.isfile(os.path.join(self.trialdir, filename)):
   plot_stim_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, area_rate)

  if self.has_spike_raster_stats:
   filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-spike-activity-area-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    spike_raster = self.get_spike_raster(neuron_type)
    plot_stim_spike_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, area_rate, spike_raster)
    
  if type(ca_rates) != type(None):
   filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-activity-ca-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    plot_stim_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, ca_rates)

   if self.has_spike_raster_stats:
    filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-spike-activity-ca-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
    if not os.path.isfile(os.path.join(self.trialdir, filename)):
     spike_raster = self.get_spike_raster(neuron_type)
     plot_stim_spike_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, ca_rates, spike_raster)

  if type(cap_rates) != type(None):
   filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-activity-no_stim-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    plot_stim_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.no_stim_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, [cap_rates[0]])

   if self.has_spike_raster_stats:
    filename = self.brain_area + '-' + neuron_type + '-' + stim_type + '-spike-activity-no_stim-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
    if not os.path.isfile(os.path.join(self.trialdir, filename)):
     spike_raster = self.get_spike_raster(neuron_type)
     plot_stim_spike_activity(self.nb_patterns, self.nb_stim, self.stim_pat, t_start, t_stop, self.time, min_rate, self.no_stim_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, stimtime, stimdata, [cap_rates[0]], spike_raster)


 def plot_spike_activity(self, neuron_type, t_start, t_stop):
  min_rate = self.get_min_rate(neuron_type)
  area_rate = self.get_area_rate(neuron_type)
  ca_rates = self.get_ca_rates(neuron_type)
  cap_rates = self.get_cap_rates(neuron_type)

  extension = '.svg'

  bin_size = self.bin_size / 60
  xlabel = 'lag (min)'
  left_xlim = None
  right_xlim = None
  skip = None

  #'''
  filename = self.brain_area + '-' + neuron_type + '-activity-area-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
  if not os.path.isfile(os.path.join(self.trialdir, filename)): 
   plot_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, area_rate)

  filename = 'activity_cross_correlation-' + neuron_type + '-' + self.brain_area + '_area-' + self.brain_area + '_area-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
  plot_activity_cross_correlation('area', 'area', 'gray', self.cell_assemb_method, self.trialdir, filename, self.brain_area, self.brain_area, neuron_type, area_rate[0], area_rate[0], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)
  
  if type(ca_rates) != type(None):
   filename = self.brain_area + '-' + neuron_type + '-activity-ca-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    plot_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, ca_rates)
    
   #for stim1, stim2 in product(range(self.nb_patterns), range(self.nb_patterns)):
   for stim1, stim2 in combinations(range(self.nb_patterns), 2):
    filename = 'activity_cross_correlation-' + neuron_type + '-' + self.brain_area + '_ca_' + str(stim1) + '-' + self.brain_area + '_ca_' + str(stim2) + '-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
    plot_activity_cross_correlation('ca_' + str(stim1), 'ca_' + str(stim2), '#FF00FF', self.cell_assemb_method, self.trialdir, filename, self.brain_area, self.brain_area, neuron_type, ca_rates[stim1], ca_rates[stim2], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

   for stim in range(self.nb_patterns):
    filename = 'activity_cross_correlation-' + neuron_type + '-' + self.brain_area + '_ca_' + str(stim) + '-' + self.brain_area + '_ca_' + str(stim) + '-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
    plot_activity_cross_correlation('ca_' + str(stim), 'ca_' + str(stim), self.colors[stim], self.cell_assemb_method, self.trialdir, filename, self.brain_area, self.brain_area, neuron_type, ca_rates[stim], ca_rates[stim], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

  
  if type(cap_rates) != type(None):
   filename = self.brain_area + '-' + neuron_type + '-activity-no_stim-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    plot_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.no_stim_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, [cap_rates[0]])

   filename = 'activity_cross_correlation-' + neuron_type + '-' + self.brain_area + '_no_stim-' + self.brain_area + '_no_stim-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
   plot_activity_cross_correlation('no_stim', 'no_stim', 'black', self.cell_assemb_method, self.trialdir, filename, self.brain_area, self.brain_area, neuron_type, cap_rates[0], cap_rates[0], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

   if type(ca_rates) != type(None):
    for stim in range(self.nb_patterns):
     filename = 'activity_cross_correlation-' + neuron_type + '-' + self.brain_area + '_no_stim-' + self.brain_area + '_ca_' + str(stim) + '-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
     plot_activity_cross_correlation('no_stim', 'ca_' + str(stim), '#FF00FF', self.cell_assemb_method, self.trialdir, filename, self.brain_area, self.brain_area, neuron_type, cap_rates[0], ca_rates[stim], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)
  #'''
  
  if self.has_spike_raster_stats:
   spike_raster = self.get_spike_raster(neuron_type)

   #'''
   filename = self.brain_area + '-' + neuron_type + '-spike-activity-area-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    plot_spike_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, area_rate, spike_raster)
   #'''

   #'''
   if type(ca_rates) != type(None):
    filename = self.brain_area + '-' + neuron_type + '-spike-activity-ca-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
    if not os.path.isfile(os.path.join(self.trialdir, filename)):
     plot_spike_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.area_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, ca_rates, spike_raster)
   #'''
   
   #'''
   if type(cap_rates) != type(None):
    filename = self.brain_area + '-' + neuron_type + '-spike-activity-no_stim-' + self.cell_assemb_method + '-' + self.full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop) + extension
    if not os.path.isfile(os.path.join(self.trialdir, filename)):
     plot_spike_activity(self.nb_patterns, t_start, t_stop, self.time, min_rate, self.no_stim_color, self.colors, self.cell_assemb_method, self.trialdir, filename, self.brain_area, neuron_type, [cap_rates[0]], spike_raster)
   #'''

 def plot_zoom_spike_stats(self, neuron_type, t_start, t_stop):
  extension = '.svg'
  
  filename = self.brain_area + '-' + neuron_type + '-spike_stats-area-' + self.full_prefix + '-t_start_' + str(t_start) + '-t_stop_' + str(t_stop) + extension
  if not os.path.isfile(os.path.join(self.trialdir, filename)):
   if not self.has_sfo(neuron_type):
    self.load_spike_view(neuron_type)
   sfo = self.get_sfo(neuron_type)
   nb_neurons = self.get_nb_neurons(neuron_type)
   zoom_spikes = sfo.get_spikes(t_start, t_stop, nb_neurons)
   plot_spike_stats(zoom_spikes, 'white', self.brain_area, neuron_type, self.trialdir, filename)


 def plot_max_isi_zoom_spike_stats(self, neuron_type, t_start, t_stop, stim_type):
  if stim_type == 'stim':
   stimtime = self.stimtime
   stimdata = self.stimdata
  elif stim_type == 'rep':
   stimtime = self.reptime
   stimdata = self.repdata
  max_t_start, max_t_stop = get_max_inter_stim_interval(stimtime, stimdata, t_start, t_stop)
  if max_t_start != None and max_t_stop != None:
   self.plot_zoom_spike_stats(neuron_type, max_t_start, max_t_stop)


 def plot_activity_zoom(self, neuron_type):
  duration = self.t_stop - self.t_start
  zr = self.zoom_range
  start_stop = [(i*zr*duration + self.t_start, (i+1)*zr*duration + self.t_start) for i in range(int(1/zr))]
  for start,stop in start_stop:
   if self.has_stim_presentation:
    self.plot_stim_spike_activity(neuron_type, start, stop, 'stim')
    if self.has_spike_stats_plots:
     self.plot_max_isi_zoom_spike_stats(neuron_type, start, stop, 'stim')
   elif self.has_rep_presentation:
    self.plot_stim_spike_activity(neuron_type, start, stop, 'rep')
    if self.has_spike_stats_plots:
     self.plot_max_isi_zoom_spike_stats(neuron_type, start, stop, 'rep')
   else:
    self.plot_spike_activity(neuron_type, start, stop)
    if self.has_spike_stats_plots:
     self.plot_zoom_spike_stats(neuron_type, start, stop)


 def plot_activity(self, neuron_type):
  if self.has_stim_presentation:
   self.plot_stim_spike_activity(neuron_type, self.t_start, self.t_stop, 'stim')
  elif self.has_rep_presentation:
   self.plot_stim_spike_activity(neuron_type, self.t_start, self.t_stop, 'rep')
  else:
   self.plot_spike_activity(neuron_type, self.t_start, self.t_stop)
  if self.zoom_range < 1:
   self.plot_activity_zoom(neuron_type)


 def store_recall_metrics(self, neuron_type, results, all_pats_metrics, ind_pats_metrics):
  results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, 'all', 'accuracy', all_pats_metrics.get_accuracy())
  results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, 'all', 'tpr', all_pats_metrics.get_tpr())
  if all_pats_metrics.has_fpr():
   results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, 'all', 'fpr', all_pats_metrics.get_fpr())
  
  for pat,ind_pat_metrics in enumerate(ind_pats_metrics):
   results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, pat, 'accuracy', ind_pat_metrics.get_accuracy())
   results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, pat, 'tpr', ind_pat_metrics.get_tpr())
   if ind_pat_metrics.has_fpr():
    results.add_result(self.trial, self.brain_area, neuron_type, self.simtime_learn, self.simtime_consolidation, self.phase, pat, 'fpr', ind_pat_metrics.get_fpr())
 

 def compute_recall_metrics(self, neuron_type, results, percentile=25):
  ###
  #for i in range(nb_stim):
  # print('stim_intervals', i, len(stim_intervals[i]))
  # print('stim_spikes', i, stim_spikes[i].shape)
  # print('rates_per_stim_presentation', i, rates_per_stim_presentation[i].shape)
  ###

  ###
  # Check computation of stimulus-evoked firing rates
  #rates_across_stim_presentations = get_rates_across_stim_presentations(nb_stim, stim_intervals, stim_spikes)
  #rates_across_stim_presentations_old = get_rates_across_stim_presentations_old(nb_stim, nb_neurons, stim_intervals, sfo)
  #rates_per_stim_presentation_old = get_rates_per_stim_presentation_old(nb_stim, nb_neurons, stim_intervals, sfo)
  #for stim in range(nb_stim):
  # print ('stim', stim)
  # print ('check o', np.sum(rates_across_stim_presentations[stim] - rates_across_stim_presentations_old[stim])) # should be 0
  # print ('check i', np.sum(rates_per_stim_presentation[stim] - rates_per_stim_presentation_old[stim])) # should be 0

  #for stim,irates in enumerate(rates_per_stim_presentation):
  # print ('stim', stim, 'ind stim rate shape', irates.shape)
  # spikes = np.zeros(irates.shape)
  # stim_duration = 0
  # for presentation in range(len(stim_intervals[stim])):
  #  stim_start, stim_stop = stim_intervals[stim][presentation]
  #  stim_duration += stim_stop - stim_start
  #  spikes[presentation, :] = irates[presentation, :] * (stim_stop - stim_start)
  # orates = np.sum(spikes, axis=0) / stim_duration
  # check = np.sum(rates_across_stim_presentations[stim] - orates)
  # print('check', check) # should be close to zero
  ###

  if not self.has_rates_per_stim(neuron_type):
   self.get_stim_stats(neuron_type) 
  rates_per_stim_presentation = self.get_rates_per_stim(neuron_type)
  cell_assemblies = self.get_cas(neuron_type)
  min_rate = self.get_min_rate(neuron_type)
  
  print("Computing stimulus-evoked %s %s cell assembly rates..."%(self.brain_area, neuron_type))
  ca_stim_rates = get_ca_stim_rates(rates_per_stim_presentation, cell_assemblies)
  #ca_stim_rates = get_ca_percentile_stim_rates(rates_per_stim_presentation, cell_assemblies, percentile)

  ###
  #ca_stim_rates_old = get_ca_stim_rates_old(self.rates_per_stim_presentation, self.cell_assemblies)
  #for r1, r2 in zip(ca_stim_rates, ca_stim_rates_old):
  # print ('check ca_stim_rates', np.sum(r1 - r2)) # should be 0
  ###

  print("Computing %s %s memory recall metrics..."%(self.brain_area, neuron_type))
  all_pats_metrics, ind_pats_metrics = get_recall_metrics(self.nb_stim, self.nb_patterns, self.stim_pat, min_rate, ca_stim_rates)

  '''
  ###
  #print('\nall_pats_metrics')
  #all_pats_metrics.print_metrics()
  ###
  all_pats_metrics.save_file(self.trialdir, self.brain_area + '-' + neuron_type + '-recall-all_pat-' + self.full_prefix + '.metrics')
  
  for pat,ind_pat_metrics in enumerate(ind_pats_metrics):
   ###
   #print('\npat', pat, 'metrics')
   #ind_pat_metrics.print_metrics()
   ###
   ind_pat_metrics.save_file(self.trialdir, self.brain_area + '-' + neuron_type + '-recall-pat_' + str(pat) + '-' + self.full_prefix + '.metrics')
  '''

  # Store metrics
  self.store_recall_metrics(neuron_type, results, all_pats_metrics, ind_pats_metrics)


 def analyze_dynamics(self, results):
  '''
  Compute and save a series of statistics to characterize network dynamics.

  List of saved files:
  - stim (*.pkl)
  - cell_assemblies (*.pat.*)
  - spikes (*.pkl)
  - activity (*.pkl)
  - metrics (*.metrics)
  '''

  self.has_dynamics_files()

  neuron_types = []
  if self.has_inh_analysis:
   neuron_types.append('inh')
  neuron_types.append('exc')

  for neuron_type in neuron_types:
   
   if self.has_stim_presentation:
    self.get_rates_across_stim_presentations(neuron_type)
   
   self.get_cell_assemblies(neuron_type)

   if self.has_rep and neuron_type == 'exc':
    self.generate_replay()
   
   self.get_activity_stats(neuron_type)

   if self.has_recall_metrics and (not self.has_recall_metrics_file) and self.has_stim_presentation and type(self.get_cas(neuron_type)) != type(None):
    self.compute_recall_metrics(neuron_type, results)

  self.delete_spike_view()


 def plot_dynamics(self):
  '''
  Plot a series of statistics to characterize network dynamics.
  '''

  neuron_types = []
  if self.has_inh_analysis:
   neuron_types.append('inh')
  neuron_types.append('exc')
  
  #neuron_types = ['exc']
  #if self.has_inh_analysis:
  # neuron_types.append('inh')

  for neuron_type in neuron_types:
   if self.has_stim_presentation and self.has_rates_across_stim_plots:
    self.plot_rates_across_stim_presentations(neuron_type)

   if self.ids_cell_assemb and self.has_cell_assemb_plots:
    self.plot_cell_assemblies(neuron_type)

   if self.has_spike_stats_plots:
    self.plot_spike_stats(neuron_type)

   if self.has_activity_plots:
    self.plot_activity(neuron_type)

   if self.has_neuron_vi_plots:   
    self.plot_vi_traces(neuron_type)


 def get_wmat(self, prefix, con, area):
  wmatfiles = ["%s/%s.%i.%s.%s.wmat"%(self.trialdir,prefix,i,con,area) for i in range(self.num_mpi_ranks)]
  return np.sum( [ mmread(wf) for wf in wmatfiles ] )


 def get_wmatrix(self, pre_post, prefix):
  weights = self.get_wmat(prefix, pre_post.connection, pre_post.get_wmat_area())
  return weights.toarray()


 def get_ordered_wmatrix(self, pre_post, prefix):
  wmatrix = self.get_wmatrix(pre_post, prefix)
  if type(pre_post.pre_ca_partitions) != type(None):
   neurons = list(range(pre_post.pre_neurons))
   partition_neurons = chain_ca_partitions(pre_post.pre_ca_partitions)
   wmatrix[neurons,:] = wmatrix[partition_neurons,:]
   ordered_pre = 'yes'
  else:
   ordered_pre = 'no'
  if type(pre_post.post_ca_partitions) != type(None):
   neurons = list(range(pre_post.post_neurons))
   partition_neurons = chain_ca_partitions(pre_post.post_ca_partitions)
   wmatrix[:,neurons] = wmatrix[:,partition_neurons]
   ordered_post = 'yes'
  else:
   ordered_post = 'no'
  return wmatrix, ordered_pre, ordered_post

 def get_mean_ca_wmatrix(self, pre_post, prefix):
  assert len(pre_post.pre_ca) == len(pre_post.post_ca),"Pre and post must have the same number of cell assemblies"
  mean_wmat = np.zeros((len(pre_post.pre_ca), len(pre_post.post_ca)))
  wmat = self.get_wmatrix(pre_post, prefix)
  for pre in range(len(pre_post.pre_ca)):
   for post in range(len(pre_post.post_ca)):
    mean_wmat[pre, post] = np.mean(wmat[pre_post.pre_ca[pre], :][:, pre_post.post_ca[post]])
  return mean_wmat

 def get_intra_inter_ca_weights(self, pre_post, prefix):
  assert len(pre_post.pre_ca) == len(pre_post.post_ca),"Pre and post must have the same number of cell assemblies"
  intra_ca_weights = []
  inter_ca_weights = []
  wmat = self.get_wmatrix(pre_post, prefix)
  for pre in range(len(pre_post.pre_ca)):
   for post in range(len(pre_post.post_ca)):
    weights = wmat[pre_post.pre_ca[pre], :][:, pre_post.post_ca[post]]
    if pre == post:
     intra_ca_weights.append(weights.flatten())
    else:
     inter_ca_weights.append(weights.flatten())
  '''
  print('intra')
  for ca in range(len(intra_ca_weights)):
   print(ca, intra_ca_weights[ca].shape)
  print('inter')
  for ca in range(len(inter_ca_weights)):
   print(ca, inter_ca_weights[ca].shape)
  '''
  return np.concatenate(intra_ca_weights), np.concatenate(inter_ca_weights)


 def get_mean_cluster_wmatrix(self, pre_post, prefix):
  wmat = self.get_wmatrix(pre_post, prefix)
  assert len(pre_post.pre_cluster) == len(pre_post.post_cluster),"Pre and post must have the same number of clusters"
  mean_wmat = np.zeros((len(pre_post.pre_cluster), len(pre_post.post_cluster)))
  for pre in range(len(pre_post.pre_cluster)):
   for post in range(len(pre_post.post_cluster)):
    mean_wmat[pre, post] = np.mean(wmat[pre_post.pre_cluster[pre], :][:, pre_post.post_cluster[post]])
  return mean_wmat


 def plot_wmat(self, pre_post, prefix):
  con_mtxfile = '-'.join([str(self.file_prefix_assemb),
                          str(self.simtime_assemb),
                          pre_post.get_wmat_area(),
                          'weight_mtx',
                          pre_post.connection,
                          prefix])
  con_mtxfile = con_mtxfile + '.svg'
  has_con_mtxfile = os.path.isfile(os.path.join(self.trialdir, con_mtxfile))
  if not has_con_mtxfile:
   wmatrix, ordered_pre, ordered_post = self.get_ordered_wmatrix(pre_post, prefix)
   plot_weight_matrix(pre_post.get_wmat_area(), pre_post.connection, wmatrix, ordered_pre, ordered_post, self.trialdir, con_mtxfile, cmap=None)


 def plot_mean_ca_wmat(self, pre_post, prefix):
  mean_con_mtxfile = '-'.join([str(self.file_prefix_assemb),
                               str(self.simtime_assemb),
                               pre_post.get_wmat_area(),
                               'weight_mtx',
                               'mean_ca',
                               pre_post.connection,
                               prefix])
  extension = '.svg'
  mean_con_mtxfile = mean_con_mtxfile + extension

  has_mean_con_mtxfile = os.path.isfile(os.path.join(self.trialdir, mean_con_mtxfile))
  if not has_mean_con_mtxfile:
   mean_wmatrix = self.get_mean_ca_wmatrix(pre_post, prefix)
   plot_mean_weight_matrix(pre_post.get_wmat_area(), pre_post.connection, mean_wmatrix, pre_post.pre_ca, pre_post.post_ca, self.trialdir, mean_con_mtxfile, 'Cell Assembly', cmap='Oranges')


 def plot_mean_cluster_wmat(self, pre_post, prefix):
  mean_con_mtxfile = '-'.join([str(self.file_prefix_assemb),
                               str(self.simtime_assemb),
                               pre_post.get_wmat_area(),
                               'weight_mtx',
                               'mean_cluster',
                               pre_post.connection,
                               prefix])
  extension = '.svg'
  mean_con_mtxfile = mean_con_mtxfile + extension
  has_mean_con_mtxfile = os.path.isfile(os.path.join(self.trialdir, mean_con_mtxfile))
  if not has_mean_con_mtxfile:
   mean_wmatrix = self.get_mean_cluster_wmatrix(pre_post, prefix)
   plot_mean_weight_matrix(pre_post.get_wmat_area(), pre_post.connection, mean_wmatrix, pre_post.pre_cluster, pre_post.post_cluster, self.trialdir, mean_con_mtxfile, 'Cluster', cmap='Oranges')


 def plot_whist(self, pre_post, prefix):
  con_histfile = '-'.join([str(self.file_prefix_assemb),
                           str(self.simtime_assemb),
                           pre_post.get_wmat_area(),
                           'weight_hist',
                           pre_post.connection,
                           prefix])
  con_histfile = con_histfile + '.svg'
  has_con_histfile = os.path.isfile(os.path.join(self.trialdir, con_histfile))
  if not has_con_histfile:
   weights = self.get_wmat(prefix, pre_post.connection, pre_post.get_wmat_area())
   plot_weight_distribution(pre_post.get_wmat_area(), pre_post.connection, weights, self.trialdir, con_histfile, color='orange')

 def convert_coordinate(self, neuron, rf_len):
  row = int(neuron / rf_len)
  col = neuron % rf_len
  return row,col


 def get_neuron_wmat(self, wmat, rf_len, neuron):
  nwmat = np.zeros((rf_len,rf_len))
  #print('nwmat', type(nwmat), nwmat.shape)
  for i,w in enumerate(wmat[:,neuron]):
   row,col = self.convert_coordinate(i, rf_len)
   #print('row', type(row), row)
   #print('col', type(col), col)
   nwmat[row,col] += w
  return nwmat


 def plot_weight_rf_hist(self, post_weight_sum, filename, color=None):
  plt.rcParams.update({'font.size': 6})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'Verdana'})
 
  #fig = plt.figure(dpi=300)
  fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
  if type(color) == type(None):
   plt.hist(post_weight_sum)
  else:
   plt.hist(post_weight_sum, color= color)
  plt.xlabel('Postsynaptic Weight Sum')
  plt.ylabel('Neurons')
  plt.savefig(os.path.join(self.trialdir, filename), format='svg', dpi=300)
  plt.close(fig)
  print('Saved plot of weight receptive field distribution')


 def plot_weight_rf_mat(self, post_wmat_sum, filename, cmap=None):
  plt.rcParams.update({'font.size': 6})
  plt.rcParams.update({'font.family': 'sans-serif'})
  plt.rcParams.update({'font.sans-serif': 'Verdana'})
  
  #fig = plt.figure(dpi=300)
  fig = plt.figure(figsize=(3.2, 2.4), dpi=300)
  #fig = plt.figure(figsize=(1.12, 0.84), dpi=300)
  
  if cmap == None:
   plt.imshow(post_wmat_sum, origin='lower')
  else:
   plt.imshow(post_wmat_sum, origin='lower', cmap=cmap)
  plt.axis('off')
  plt.colorbar()
  plt.savefig(os.path.join(self.trialdir, filename), format='svg', dpi=300)
  plt.close(fig)
  print('Saved plot of weight receptive field matrix')


 def plot_weight_receptive_field(self, wmatrix, rfmfile, rfhfile):
  rf_len = int(math.sqrt(wmatrix.shape[0]))
  neuron_weight_sum = []
  all_nwmat = np.zeros((rf_len,rf_len))
  for neuron in range(wmatrix.shape[1]):
   nwmat = self.get_neuron_wmat(wmatrix, rf_len, neuron)
   neuron_weight_sum.append(np.sum(nwmat))
   all_nwmat += nwmat
  self.plot_weight_rf_mat(all_nwmat, rfmfile)
  self.plot_weight_rf_hist(neuron_weight_sum, rfhfile, color='orange')


 def plot_wrf(self, pre_post, prefix):
  rfmfile = '-'.join([str(self.file_prefix_assemb),
                      str(self.simtime_assemb),
                      pre_post.get_wmat_area(),
                      'weight_mtx',
                      'rf',
                      pre_post.connection,
                      prefix])
  rfmfile = rfmfile + '.svg'
  rfhfile = '-'.join([str(self.file_prefix_assemb),
                      str(self.simtime_assemb),
                      pre_post.get_wmat_area(),
                      'weight_hist',
                      'rf',
                      pre_post.connection,
                      prefix])
  rfhfile = rfhfile + '.svg'
  has_rf_files = os.path.isfile(os.path.join(self.trialdir, rfmfile)) and os.path.isfile(os.path.join(self.trialdir, rfhfile))
  if not has_rf_files:
   wmatrix = self.get_wmatrix(pre_post, prefix)
   self.plot_weight_receptive_field(wmatrix, rfmfile, rfhfile)


 def plot_intra_inter_ca_weights(self, pre_post, chosen_prefix, intra_ca, inter_ca):
  #intra_ca, inter_ca = self.get_intra_inter_ca_weights(pre_post, chosen_prefix)
  basefilename = '-'.join(['intra_inter_ca_weights','cdf',pre_post.get_wmat_area(),pre_post.connection, chosen_prefix])
  extension = '.svg'
  filename = basefilename + extension
  plot_comp_weight_cdf(pre_post.get_wmat_area(), pre_post.connection, intra_ca, inter_ca, 'intra_ca', 'inter_ca', 'gray', 'black', 'weights', self.trialdir, filename, log=True)
  


 def get_rec_ee_pre_post(self):
  pre_neurons = self.nb_exc_neurons
  post_neurons = self.nb_exc_neurons
  if hasattr(self, 'exc_ca_partitions') and hasattr(self, 'exc_cell_assemblies'):
   pre_ca_partitions = self.exc_ca_partitions
   post_ca_partitions = self.exc_ca_partitions
   pre_ca = self.exc_cell_assemblies
   post_ca = self.exc_cell_assemblies
   if hasattr(self, 'exc_clusters'):
    pre_cluster = self.exc_clusters
    post_cluster = self.exc_clusters
   else:
    pre_cluster = None
    post_cluster = None
  else:
   pre_ca_partitions = None
   post_ca_partitions = None
   pre_ca = None
   post_ca = None
   pre_cluster = None
   post_cluster = None
  return PrePost('ee', self.brain_area, self.brain_area, pre_neurons, post_neurons, pre_ca_partitions, post_ca_partitions, pre_ca, post_ca, pre_cluster, post_cluster)


 def get_rec_ie_pre_post(self):
  pre_neurons = self.nb_inh_neurons
  post_neurons = self.nb_exc_neurons
  
  if hasattr(self, 'inh_ca_partitions') and hasattr(self, 'inh_cell_assemblies'):
   pre_ca_partitions = self.inh_ca_partitions
   pre_ca = self.inh_cell_assemblies
   if hasattr(self, 'inh_clusters'):
    pre_cluster = self.inh_clusters
   else:
    pre_cluster = None
  else:
   pre_ca_partitions = None
   pre_ca = None
   pre_cluster = None
  
  if hasattr(self, 'exc_ca_partitions') and hasattr(self, 'exc_cell_assemblies'):
   post_ca_partitions = self.exc_ca_partitions
   post_ca = self.exc_cell_assemblies
   if hasattr(self, 'exc_clusters'):
    post_cluster = self.exc_clusters
   else:
    post_cluster = None 
  else:
   post_ca_partitions = None
   post_ca = None
   post_cluster = None
  return PrePost('ie', self.brain_area, self.brain_area, pre_neurons, post_neurons, pre_ca_partitions, post_ca_partitions, pre_ca, post_ca, pre_cluster, post_cluster)


 def get_recurrent_pre_post(self, connection):
  assert connection == 'ee' or connection == 'ie',"Only plots of ee and ie weights are enabled for recurrent connections"
  if connection == 'ee':
   return self.get_rec_ee_pre_post()
  elif connection == 'ie':
   return self.get_rec_ie_pre_post()


 def get_stim_area_pre_post(self, connection):
  assert connection == 'ee',"Only plots of ee weights are enabled for stim->brain_area"
  pre_neurons = self.nb_exc_neurons
  post_neurons = self.nb_exc_neurons
  pre_ca_partitions = self.stim_partitions
  pre_ca = self.stim_pats
  pre_cluster = self.stim_clusters
  if hasattr(self, 'exc_ca_partitions') and hasattr(self, 'exc_cell_assemblies'):
   post_ca_partitions = self.exc_ca_partitions
   post_ca = self.exc_cell_assemblies
   if hasattr(self, 'exc_clusters'):
    post_cluster = self.exc_clusters
   else:
    post_cluster = None
  else:
   post_ca_partitions = None
   post_ca = None
   post_cluster = None
  return PrePost('ee', 'stim', self.brain_area, pre_neurons, post_neurons, pre_ca_partitions, post_ca_partitions, pre_ca, post_ca, pre_cluster, post_cluster)
  

 def get_area_area_pre_post(self, connection, pre_area, post_area, pre_area_ana):
  assert connection == 'ee',"Only plots of ee weights are enabled for brain_area->brain_area"
  pre_neurons = pre_area_ana.nb_exc_neurons
  post_neurons = self.nb_exc_neurons
  if hasattr(pre_area_ana, 'exc_ca_partitions') and hasattr(pre_area_ana, 'exc_cell_assemblies'):
   pre_ca_partitions = pre_area_ana.exc_ca_partitions
   pre_ca = pre_area_ana.exc_cell_assemblies
   if hasattr(pre_area_ana, 'exc_clusters'):
    pre_cluster = pre_area_ana.exc_clusters
   else:
    pre_cluster = None
  else:
   pre_ca_partitions = None
   pre_ca = None
   pre_cluster = None
  if hasattr(self, 'exc_ca_partitions') and hasattr(self, 'exc_cell_assemblies'):
   post_ca_partitions = self.exc_ca_partitions
   post_ca = self.exc_cell_assemblies
   if hasattr(self, 'exc_clusters'):
    post_cluster = self.exc_clusters
   else:
    post_cluster = None
  else:
   post_ca_partitions = None
   post_ca = None
   post_cluster = None
  return PrePost('ee', pre_area, post_area, pre_neurons, post_neurons, pre_ca_partitions, post_ca_partitions, pre_ca, post_ca, pre_cluster, post_cluster)


 def get_pre_post(self, pre_area, post_area, connection, pre_area_ana=None):
  if pre_area == post_area:
   return self.get_recurrent_pre_post(connection)
  elif pre_area == 'stim' and post_area == self.brain_area:
   return self.get_stim_area_pre_post(connection)
  elif pre_area in self.other_areas and post_area == self.brain_area:
   return self.get_area_area_pre_post(connection, pre_area, post_area, pre_area_ana)
  else:
   return PrePost(connection, pre_area, post_area)

 
 def get_cluster_incoming_weights(self, pre_area, post_area, connection, prefix, pre_area_ana=None):
  '''
  Extract sum of weights to neurons of each cluster for specified connection and time point.
  '''
  incoming_weights = []
  pre_post = self.get_pre_post(pre_area, post_area, connection, pre_area_ana=pre_area_ana)
  wmatrix = self.get_wmatrix(pre_post, prefix)
  for cluster in pre_post.post_cluster:
   incoming_weights.append(np.sum(wmatrix[:,cluster], axis=0))
  return incoming_weights


 def get_ca_incoming_weights(self, pre_area, post_area, connection, prefix, pre_area_ana=None):
  '''
  Extract sum of weights to engram cells for specified connection and time point.
  '''
  #incoming_weights = []
  pre_post = self.get_pre_post(pre_area, post_area, connection, pre_area_ana=pre_area_ana)
  wmatrix = self.get_wmatrix(pre_post, prefix)
  cas =[]
  for ca in pre_post.post_ca:
   cas.extend(ca)
  cas = list(set(cas))
  #incoming_weights.append(np.sum(wmatrix[:,cas], axis=0))
  #return incoming_weights
  return np.sum(wmatrix[:,cas], axis=0)


 def get_rec_incoming_weights(self, prefix):
  self.ee_incoming_weights = self.get_cluster_incoming_weights(self.brain_area, self.brain_area, 'ee', prefix)
  #print('ee start')
  #self.print_incoming_weights()
  self.add_incoming_weights(self.ee_incoming_weights)
  #print('ee end')
  #self.print_incoming_weights()

  self.ie_incoming_weights = self.get_cluster_incoming_weights(self.brain_area, self.brain_area, 'ie', prefix)
  #print('ie start')
  #self.print_incoming_weights()
  self.add_incoming_weights(self.ie_incoming_weights)
  #print('ie end')
  #self.print_incoming_weights()


 def get_stim_area_incoming_weights(self, prefix):
  self.stim_area_incoming_weights = self.get_cluster_incoming_weights('stim', self.brain_area, 'ee', prefix)
  #print('stim_area start')
  #self.print_incoming_weights()
  self.add_incoming_weights(self.stim_area_incoming_weights)
  #print('stim_area end')
  #self.print_incoming_weights()


 def get_area_area_incoming_weights(self, prefix, pre_area, pre_area_ana):
  self.area_area_incoming_weights.append(self.get_cluster_incoming_weights(pre_area, self.brain_area, 'ee', prefix, pre_area_ana=pre_area_ana))
  #print('area_area start')
  #self.print_incoming_weights()
  self.add_incoming_weights(self.area_area_incoming_weights[-1])
  #print('area_area end')
  #self.print_incoming_weights()

 def store_weight_metrics(self, results, pre_post, chosen_prefix, intra_ca, inter_ca):

  #results.add_result(self.trial, pre_post.get_wmat_area(), pre_post.connection, self.simtime_learn, self.simtime_consolidation, chosen_prefix, 'all', 'intra_ca_weight', np.mean(intra_ca))
  
  #results.add_result(self.trial, pre_post.get_wmat_area(), pre_post.connection, self.simtime_learn, self.simtime_consolidation, self.chosen_prefix, 'all', 'inter_ca_weight', np.mean(inter_ca))
  
  _, ks_pvalue = kstest(intra_ca, inter_ca, alternative='two-sided', mode='auto')
  
  results.add_result(self.trial, pre_post.get_wmat_area(), pre_post.connection, self.simtime_learn, self.simtime_consolidation, chosen_prefix, 'all', 'intra_inter_ca_weight_ks_pvalue', ks_pvalue)


 def plot_rec_weights(self, chosen_prefix, compute_weight_metrics, results):
  for connection in ['ee','ie']:
   pre_post = self.get_pre_post(self.brain_area, self.brain_area, connection)

   if compute_weight_metrics:
    print('Computing %s %s weight metrics'%(pre_post.get_wmat_area(), pre_post.connection))
    if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
     intra_ca, inter_ca = self.get_intra_inter_ca_weights(pre_post, chosen_prefix)
     self.store_weight_metrics(results, pre_post, chosen_prefix, intra_ca, inter_ca)
     if self.has_metrics_plots:
      self.plot_intra_inter_ca_weights(pre_post, chosen_prefix, intra_ca, inter_ca)

   else:
    self.plot_wmat(pre_post, chosen_prefix)
    if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
     self.plot_mean_ca_wmat(pre_post, chosen_prefix)
    if (type(pre_post.pre_cluster) != type(None)) and (type(pre_post.post_cluster) != type(None)):
     self.plot_mean_cluster_wmat(pre_post, chosen_prefix)
    self.plot_whist(pre_post, chosen_prefix)
    
   


 def plot_stim_brain_area_weights(self, chosen_prefix, compute_weight_metrics, results):
  pre_post = self.get_pre_post('stim', self.brain_area, 'ee')

  if compute_weight_metrics:
   print('Computing %s %s weight metrics'%(pre_post.get_wmat_area(), pre_post.connection))
   if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
    intra_ca, inter_ca = self.get_intra_inter_ca_weights(pre_post, chosen_prefix)
    self.store_weight_metrics(results, pre_post, chosen_prefix, intra_ca, inter_ca)
    if self.has_metrics_plots:
     self.plot_intra_inter_ca_weights(pre_post, chosen_prefix, intra_ca, inter_ca)

  else:
   self.plot_wmat(pre_post, chosen_prefix)
   if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
    self.plot_mean_ca_wmat(pre_post, chosen_prefix)
   if (type(pre_post.pre_cluster) != type(None)) and (type(pre_post.post_cluster) != type(None)):
    self.plot_mean_cluster_wmat(pre_post, chosen_prefix)
   self.plot_whist(pre_post, chosen_prefix)
   self.plot_wrf(pre_post, chosen_prefix)


 def plot_area_area_weights(self, chosen_prefix, other_area, other_area_ana, compute_weight_metrics, results):
  pre_post = self.get_pre_post(other_area, self.brain_area, 'ee', pre_area_ana=other_area_ana)

  if compute_weight_metrics:
   print('Computing %s %s weight metrics'%(pre_post.get_wmat_area(), pre_post.connection))
   if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
    intra_ca, inter_ca = self.get_intra_inter_ca_weights(pre_post, chosen_prefix)
    self.store_weight_metrics(results, pre_post, chosen_prefix, intra_ca, inter_ca)
    if self.has_metrics_plots:
     self.plot_intra_inter_ca_weights(pre_post, chosen_prefix, intra_ca, inter_ca)

  else:
   self.plot_wmat(pre_post, chosen_prefix)
   if (type(pre_post.pre_ca) != type(None)) and (type(pre_post.post_ca) != type(None)):
    self.plot_mean_ca_wmat(pre_post, chosen_prefix)
   if (type(pre_post.pre_cluster) != type(None)) and (type(pre_post.post_cluster) != type(None)):
    self.plot_mean_cluster_wmat(pre_post, chosen_prefix)
   self.plot_whist(pre_post, chosen_prefix)


 def plot_weights(self,
                  chosen_prefix,
                  has_rec_plots=True,
                  has_stim_brain_area_plots=True,
                  has_area_area_plots=False,
                  other_area=None,
                  other_area_ana=None,
                  compute_weight_metrics=False,
                  results=None):
  #'''
  if has_rec_plots:
   self.plot_rec_weights(chosen_prefix, compute_weight_metrics, results)
  #'''
  
  #'''
  if self.has_stim_brain_area and has_stim_brain_area_plots:
   self.plot_stim_brain_area_weights(chosen_prefix, compute_weight_metrics, results)
  #'''

  #'''
  if (type(other_area) != type(None)) and (type(other_area_ana) != type(None)):
   other_area_idx = self.other_areas.index(other_area)
   has_area_area = self.has_area_area[other_area_idx]
   if has_area_area and has_area_area_plots:
    self.plot_area_area_weights(chosen_prefix, other_area, other_area_ana, compute_weight_metrics, results)
  #'''

 def add_incoming_weights(self, con_incoming_weights):
  assert len(self.incoming_weights) == len(con_incoming_weights),"Number of clusters must match"
  for i in range(len(self.incoming_weights)):
   self.incoming_weights[i] += con_incoming_weights[i]


 def print_incoming_weights(self):
  print('\nincoming_weights')
  for c,iw in enumerate(self.incoming_weights):
   print('cluster', c, 'shape', iw.shape, 'sum', np.sum(iw))


 def get_in_weights(self, chosen_prefix, other_area_anas):
  self.incoming_weights = [np.zeros(len(c)) for c in self.exc_clusters]
  
  #self.print_incoming_weights()
  
  self.get_rec_incoming_weights(chosen_prefix)
  
  if self.has_stim_brain_area:
   self.get_stim_area_incoming_weights(chosen_prefix)
   
  self.area_area_incoming_weights = []
  for other_area,other_area_ana,has_area_area in zip(self.other_areas,other_area_anas,self.has_area_area):
   #if self.has_area_area and (self.file_prefix_hm not in chosen_prefix):
   if has_area_area and (type(other_area_ana) != type(None)):
    self.get_area_area_incoming_weights(chosen_prefix, other_area, other_area_ana)
   else:
    self.area_area_incoming_weights.append(None)


 def plot_cluster_incoming_weights(self, chosen_prefix, other_area_anas):
  '''
  Plot sum of incoming weights to neurons of every excitatory cluster by projection and in total.
  '''
  basefile = '-'.join([str(self.file_prefix_assemb),
                       str(self.simtime_assemb),
                       self.brain_area,
                       'incoming_weights',
                       'cdf'])

  cons = ['ee', 'ie']
  if self.has_stim_brain_area:
   cons.append('stim_' + self.brain_area)
  for other_area,other_area_ana,has_area_area in zip(self.other_areas,other_area_anas,self.has_area_area):
   #if has_area_area and (self.file_prefix_hm not in chosen_prefix):
   if has_area_area and (type(other_area_ana) != type(None)):
    cons.append(other_area + '_' + self.brain_area)

  has_plots = True
  for con in cons:
   filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
   if not os.path.isfile(os.path.join(self.trialdir, filename)):
    has_plots = False
    break

  if not has_plots:
   #self.print_identifier()
   self.get_in_weights(chosen_prefix, other_area_anas)
   
   con = 'ee'
   filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
   plot_weight_cdf(self.brain_area, con, self.ee_incoming_weights, self.colors, self.trialdir, filename)

   con = 'ie'
   filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
   plot_weight_cdf(self.brain_area, con, self.ie_incoming_weights, self.colors, self.trialdir, filename)

   if self.has_stim_brain_area:
    con = 'stim_' + self.brain_area
    filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
    plot_weight_cdf(self.brain_area, con, self.stim_area_incoming_weights, self.colors, self.trialdir, filename)

   for other_area,other_area_ana,has_area_area,area_area_incoming_weights in zip(self.other_areas,other_area_anas,self.has_area_area,self.area_area_incoming_weights):
    #if self.has_area_area and (self.file_prefix_hm not in chosen_prefix):
    if has_area_area and (type(other_area_ana) != type(None)):
     con = other_area + '_' + self.brain_area
     filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
     plot_weight_cdf(self.brain_area, con, area_area_incoming_weights, self.colors, self.trialdir, filename)

   con = 'total'
   filename = basefile + '-' + con + '-' + chosen_prefix + '.svg'
   plot_weight_cdf(self.brain_area, con, self.incoming_weights, self.colors, self.trialdir, filename)

  
 def print_identifier(self):
  print('\nbrain_area: %s'%self.brain_area)
  print("full_prefix: %s"%self.full_prefix)
  print("prefix: %s"%self.prefix)
  print("phase: %s"%self.phase)
  print("simtime_learn:", self.simtime_learn)
  print("simtime_consolidation:", self.simtime_consolidation)
  print("random_trial: %s"%self.trial)
  print("trialdir: %s"%self.trialdir)


 def analyze(self, recall_results, weight_results):
  print("\n\n\nRunning Analyzer:")
  self.print_identifier()

  if self.simtime > 0:
   
   self.analyze_dynamics(recall_results)

   if self.has_weight_metrics and (not self.has_weight_metrics_file):
    self.plot_weights(self.full_prefix, compute_weight_metrics=True, results=weight_results)
    if self.phase in ['burn-in']:
     self.plot_weights(self.prefix, compute_weight_metrics=True, results=weight_results)
   
   if self.has_plots:
    self.plot_dynamics()
    if self.has_weight_plots:
     self.plot_weights(self.full_prefix)
     if self.phase in ['burn-in']:
      self.plot_weights(self.prefix)

