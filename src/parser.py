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
import pandas as pd
import numpy as np
from scipy.stats import kstest

from analyzer import Analyzer
from results import Results
from helper import *

class Parser:
 def __init__(self, run, has_inh_analysis, has_recall_metrics, has_weight_metrics, has_plots, has_neuron_vi_plots, has_cell_assemb_plots, has_activity_stats, has_activity_plots, has_spike_raster_stats, has_spike_stats_plots, has_metrics_plots, has_rates_across_stim_plots, has_weight_plots, has_rf_plots, conf_int, num_mpi_ranks, integration_time_step, rundir, random_trials, brain_areas, prefixes_thl, prefixes_ctx, prefixes_hpc, prefixes_rdt, file_prefix_burn, file_prefix_learn, file_prefix_learn_bkw, file_prefix_consolidation, file_prefix_probe, file_prefix_cue, file_prefix_hm, file_prefix_burn_hm, file_prefix_learn_hm, file_prefix_consolidation_hm, file_prefix_probe_hm, file_prefix_cue_hm, phase_burn, phase_learn, phase_consolidation, phase_probe, phase_cue, nb_stim_learn, nb_stim_probe, nb_stim_cue, nb_exc_neurons_thl, exc_inh_thl, nb_exc_neurons_ctx, exc_inh_ctx, nb_exc_neurons_hpc, exc_inh_hpc, nb_exc_neurons_rdt, exc_inh_rdt, nb_neurons_stim_replay, nb_neuron_cons_stim_replay, nb_patterns, simtime_burn, simtimes_learn, simtime_learn_bkw, simtimes_consolidation, simtime_probe, simtime_cue, range_burn, range_learn, range_consolidation, range_probe, range_cue, zoom_range, nb_plot_neurons, colors, bin_size, has_stim_hpc, has_stim_ctx, has_stim_thl, has_stim_rdt, has_thl_ctx, has_ctx_thl, has_thl_hpc, has_hpc_thl, has_ctx_hpc, has_hpc_ctx, has_thl_rdt, has_ctx_rdt, has_hpc_rdt, has_rdt_thl, has_rdt_ctx, has_rdt_hpc, has_hpc_rep, ids_cell_assemb, cell_assemb_method, exc_min_rate, inh_min_rate, min_weight, stim_pat_learn, stim_pat_probe, stim_pat_cue, exc_record_rank_thl, inh_record_rank_thl, exc_ampa_nmda_ratio_thl, inh_ampa_nmda_ratio_thl, exc_record_rank_ctx, inh_record_rank_ctx, exc_ampa_nmda_ratio_ctx, inh_ampa_nmda_ratio_ctx, exc_record_rank_hpc, inh_record_rank_hpc, exc_ampa_nmda_ratio_hpc, inh_ampa_nmda_ratio_hpc, u_rest, u_exc, u_inh, stim):

  self.run = run
  self.has_inh_analysis = has_inh_analysis
  self.has_recall_metrics = has_recall_metrics
  self.has_weight_metrics = has_weight_metrics
  self.has_plots = has_plots
  self.has_neuron_vi_plots = has_neuron_vi_plots
  self.has_cell_assemb_plots = has_cell_assemb_plots
  self.has_activity_stats = has_activity_stats
  self.has_activity_plots = has_activity_plots
  self.has_spike_raster_stats = has_spike_raster_stats
  self.has_spike_stats_plots = has_spike_stats_plots
  self.has_metrics_plots = has_metrics_plots
  self.has_rates_across_stim_plots = has_rates_across_stim_plots
  self.has_weight_plots = has_weight_plots
  self.has_rf_plots = has_rf_plots
  self.conf_int = conf_int
  self.num_mpi_ranks = num_mpi_ranks
  self.integration_time_step = integration_time_step
  self.rundir = rundir
  self.random_trials = random_trials
  self.brain_areas = brain_areas
  self.prefixes_thl = prefixes_thl
  self.prefixes_ctx = prefixes_ctx
  self.prefixes_hpc = prefixes_hpc
  self.prefixes_rdt = prefixes_rdt
  self.file_prefix_burn = file_prefix_burn
  self.file_prefix_learn = file_prefix_learn
  self.file_prefix_learn_bkw = file_prefix_learn_bkw
  self.file_prefix_consolidation = file_prefix_consolidation
  self.file_prefix_probe = file_prefix_probe
  self.file_prefix_cue = file_prefix_cue
  self.file_prefix_hm = file_prefix_hm
  self.file_prefix_burn_hm = file_prefix_burn_hm
  self.file_prefix_learn_hm = file_prefix_learn_hm
  self.file_prefix_consolidation_hm = file_prefix_consolidation_hm
  self.file_prefix_probe_hm = file_prefix_probe_hm
  self.file_prefix_cue_hm = file_prefix_cue_hm
  self.phase_burn = phase_burn  
  self.phase_learn = phase_learn
  self.phase_consolidation = phase_consolidation
  self.phase_probe = phase_probe
  self.phase_cue = phase_cue
  self.nb_stim_learn = nb_stim_learn
  self.nb_stim_probe = nb_stim_probe
  self.nb_stim_cue = nb_stim_cue
  self.nb_exc_neurons_thl = nb_exc_neurons_thl
  self.nb_inh_neurons_thl = int(self.nb_exc_neurons_thl / exc_inh_thl)
  self.nb_exc_neurons_ctx = nb_exc_neurons_ctx
  self.nb_inh_neurons_ctx = int(self.nb_exc_neurons_ctx / exc_inh_ctx)
  self.nb_exc_neurons_hpc = nb_exc_neurons_hpc
  self.nb_inh_neurons_hpc = int(self.nb_exc_neurons_hpc / exc_inh_hpc)
  self.nb_exc_neurons_rdt = nb_exc_neurons_rdt
  self.nb_inh_neurons_rdt = int(self.nb_exc_neurons_rdt / exc_inh_rdt)
  self.nb_neurons_stim_replay = nb_neurons_stim_replay
  self.nb_neuron_cons_stim_replay = nb_neuron_cons_stim_replay
  self.nb_patterns = nb_patterns
  self.simtime_burn = simtime_burn
  self.simtimes_learn = simtimes_learn
  self.simtime_learn_bkw = simtime_learn_bkw
  self.simtimes_consolidation = simtimes_consolidation
  self.simtime_probe = simtime_probe
  self.simtime_cue = simtime_cue
  self.range_burn = range_burn
  self.range_learn = range_learn
  self.range_consolidation = range_consolidation
  self.range_probe = range_probe
  self.range_cue = range_cue
  self.zoom_range = zoom_range
  self.nb_plot_neurons = nb_plot_neurons
  self.colors = colors
  self.bin_size = bin_size
  self.has_stim_hpc = has_stim_hpc
  self.has_stim_ctx = has_stim_ctx
  self.has_stim_thl = has_stim_thl
  self.has_stim_rdt = has_stim_rdt
  self.has_thl_ctx = has_thl_ctx
  self.has_ctx_thl = has_ctx_thl
  self.has_thl_hpc = has_thl_hpc
  self.has_hpc_thl = has_hpc_thl
  self.has_ctx_hpc = has_ctx_hpc
  self.has_hpc_ctx = has_hpc_ctx
  self.has_thl_rdt = has_thl_rdt
  self.has_ctx_rdt = has_ctx_rdt
  self.has_hpc_rdt = has_hpc_rdt
  self.has_rdt_thl = has_rdt_thl
  self.has_rdt_ctx = has_rdt_ctx
  self.has_rdt_hpc = has_rdt_hpc
  self.has_hpc_rep = has_hpc_rep
  self.ids_cell_assemb = ids_cell_assemb
  self.cell_assemb_method = cell_assemb_method
  self.exc_min_rate = exc_min_rate
  self.inh_min_rate = inh_min_rate
  self.min_weight = min_weight
  self.stim_pat_learn = stim_pat_learn
  self.stim_pat_probe = stim_pat_probe
  self.stim_pat_cue = stim_pat_cue
  self.exc_record_rank_thl = exc_record_rank_thl
  self.inh_record_rank_thl = inh_record_rank_thl
  self.exc_ampa_nmda_ratio_thl = exc_ampa_nmda_ratio_thl
  self.inh_ampa_nmda_ratio_thl = inh_ampa_nmda_ratio_thl
  self.exc_record_rank_ctx = exc_record_rank_ctx
  self.inh_record_rank_ctx = inh_record_rank_ctx
  self.exc_ampa_nmda_ratio_ctx = exc_ampa_nmda_ratio_ctx
  self.inh_ampa_nmda_ratio_ctx = inh_ampa_nmda_ratio_ctx
  self.exc_record_rank_hpc = exc_record_rank_hpc
  self.inh_record_rank_hpc = inh_record_rank_hpc
  self.exc_ampa_nmda_ratio_hpc = exc_ampa_nmda_ratio_hpc
  self.inh_ampa_nmda_ratio_hpc = inh_ampa_nmda_ratio_hpc
  self.u_rest = u_rest
  self.u_exc = u_exc
  self.u_inh = u_inh
  self.stim = stim

  self.recall_results = Results()
  self.weight_results = Results()
  assert len(self.random_trials) > 0,"You must supply at least one trial."
  if len(self.random_trials) == 1:
   self.analyzers = []
   self.analyzers_ids = {}
   self.full_prefixes_thl = []
   self.full_prefixes_ctx = []
   self.full_prefixes_hpc = []
   self.full_prefixes_rdt = []
   self.recall_metrics_filename = 'metrics-recall-' + str(self.random_trials[0]) + '.csv'
   self.has_recall_metrics_file = os.path.isfile(os.path.join(self.rundir, self.recall_metrics_filename))
   self.weight_metrics_filename = 'metrics-weight-' + str(self.random_trials[0]) + '.csv'
   self.has_weight_metrics_file = os.path.isfile(os.path.join(self.rundir, self.weight_metrics_filename))
   self.parse()
  else:
   self.recall_metrics_filename = 'metrics-recall-all.csv'
   self.has_recall_metrics_file = os.path.isfile(os.path.join(self.rundir, self.recall_metrics_filename))
   self.weight_metrics_filename = 'metrics-weight-all.csv'
   self.has_weight_metrics_file = os.path.isfile(os.path.join(self.rundir, self.weight_metrics_filename))


 def get_analyzer_id(self, brain_area, full_prefix):
  return brain_area + '-' + full_prefix
 

 def get_analyzer(self, brain_area, full_prefix):
  analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
  if analyzer_id in self.analyzers_ids:
   analyzer_idx = self.analyzers_ids[analyzer_id]
   return self.analyzers[analyzer_idx]
  else:
   return None


 def get_phase_hm(self, phase):
  return phase + '-' + self.file_prefix_hm

 
 def get_phase_bkw(self, phase):
  return phase + '-bkw'

 
 def parse(self):
  if self.run == 1:
   self.parse_run_1()
   
  elif self.run == 2 or self.run == 3:
   print("TO-DO: Update parse_run_2_3")
   sys.exit()
   #self.parse_run_2_3()
   
  elif self.run == 4:
   print("run 4 is deprecated")
   sys.exit()
   #self.parse_run_4()
   
  elif self.run == 5 or self.run == 6:
   self.parse_run_5_6()
   
  elif self.run == 7 or self.run == 8:
   print("TO-DO: Update parse_run_7_8")
   sys.exit()
   #self.parse_run_7_8()

   
 def parse_run_1(self):

  file_prefix_assemb = self.file_prefix_learn
  has_rep = False
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     other_areas = ['ctx', 'hpc', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl, self.has_rdt_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    elif brain_area == 'ctx':
     other_areas = ['hpc', 'thl', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx, self.has_rdt_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_areas = ['ctx', 'thl', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc, self.has_rdt_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    elif brain_area == 'rdt':
     other_areas = ['thl', 'ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_rdt
     nb_inh_neurons = self.nb_inh_neurons_rdt
     has_stim_brain_area = self.has_stim_rdt
     has_area_area = [self.has_thl_rdt, self.has_ctx_rdt, self.has_hpc_rdt]
     prefixes = self.prefixes_rdt
     full_prefixes = self.full_prefixes_rdt
     exc_record_rank = None
     inh_record_rank = None
     exc_ampa_nmda_ratio = None
     inh_ampa_nmda_ratio = None
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     simtime_assemb = simtime_learn

     for prefix_idx in range(len(prefixes)):
      prefix = prefixes[prefix_idx]
      if brain_area == 'hpc' and self.file_prefix_hm in prefix:
       continue
      full_prefix = prefix
      if prefix == self.file_prefix_burn:
       full_prefix += '-' + str(self.simtime_burn)
       ids_cell_assemb = False
       has_stim_presentation = False
       simtime = self.simtime_burn
       t_stop = self.simtime_burn
       t_start = simtime - self.range_burn
       nb_stim = None
       stim_pat = None
       phase = self.phase_burn       
      else:
       if prefix == self.file_prefix_learn:
        prefix_end = '-' + str(self.simtime_burn)
        full_prefix += prefix_end + '-' + str(simtime_learn)
        ids_cell_assemb = True        
        has_stim_presentation = True
        simtime = simtime_learn
        t_stop = simtime_learn
        t_start = simtime - self.range_learn
        nb_stim = self.nb_stim_learn
        stim_pat = self.stim_pat_learn
        phase = self.phase_learn
       elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
        prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
        full_prefix += prefix_end + '-' + str(self.simtime_cue)
        ids_cell_assemb = False
        has_stim_presentation = True
        simtime = self.simtime_cue
        t_stop = self.simtime_cue
        t_start = simtime - self.range_cue
        nb_stim = self.nb_stim_cue
        stim_pat = self.stim_pat_cue        
        if prefix == self.file_prefix_cue_hm:
         phase = self.get_phase_hm(self.phase_cue)
        else:
         phase = self.phase_cue
       prefix += prefix_end
      
      analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
      if analyzer_id not in self.analyzers_ids:
       full_prefixes.append(full_prefix)

       if has_stim_presentation:
        stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
       else:
        stimtime, stimdata = None, None

       self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.min_weight, self.has_inh_analysis, self.has_recall_metrics, self.has_weight_metrics, self.has_plots, self.has_metrics_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_recall_metrics_file, self.has_weight_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, stim=self.stim))

       self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
       print("\n\n\nParsed Analyzer:")
       self.analyzers[-1].print_identifier()

        
 def parse_run_2_3(self):

  #prefixes_ctx = [self.file_prefix_burn, self.file_prefix_learn, self.file_prefix_consolidation, self.file_prefix_probe, self.file_prefix_cue]

  if self.run == 2:
   has_rep = False
  elif self.run == 3:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    if brain_area == 'ctx':
     file_prefix_assemb = self.file_prefix_probe
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      if brain_area == 'ctx':
       simtime_assemb = simtime_consolidation
      elif brain_area in ['hpc','thl']:
       simtime_assemb = simtime_learn

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         if brain_area in ['hpc','thl']:
          ids_cell_assemb = True
         elif brain_area == 'ctx':
          ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         if brain_area in ['hpc','thl']:
          ids_cell_assemb = False
         elif brain_area == 'ctx':
          ids_cell_assemb = True
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation) + '-' + str(self.simtime_probe)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, simtime_consolidation, reptime, repdata, has_rep_presentation))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_4(self):

  has_rep = False
  has_rep_presentation = False
  file_prefix_assemb = self.file_prefix_probe
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'ctx':
     other_area = 'hpc'
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = self.has_hpc_ctx
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     other_area = 'ctx'
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = self.has_ctx_hpc
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:

      simtime_assemb = str(simtime_learn) + '-' + str(simtime_consolidation)

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_learn_bkw:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(self.simtime_learn_bkw)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = self.simtime_learn_bkw
         t_stop = self.simtime_learn_bkw
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.get_phase_bkw(self.phase_learn)
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         ids_cell_assemb = True
         has_stim_presentation = True
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(self.simtime_learn_bkw) + '-' + str(simtime_consolidation) + '-' + str(self.simtime_probe)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end
       full_prefixes.append(full_prefix)

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
        
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, simtime_consolidation, reptime, repdata, has_rep_presentation))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_5_6(self):

  if self.run == 5:
   has_rep = False
  elif self.run == 6:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'hpc', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl, self.has_rdt_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    elif brain_area == 'ctx':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['hpc', 'thl', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx, self.has_rdt_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['ctx', 'thl', 'rdt']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc, self.has_rdt_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    elif brain_area == 'rdt':
     file_prefix_assemb = self.file_prefix_learn
     other_areas = ['thl', 'ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_rdt
     nb_inh_neurons = self.nb_inh_neurons_rdt
     has_stim_brain_area = self.has_stim_rdt
     has_area_area = [self.has_thl_rdt, self.has_ctx_rdt, self.has_hpc_rdt]
     prefixes = self.prefixes_rdt
     full_prefixes = self.full_prefixes_rdt
     exc_record_rank = None
     inh_record_rank = None
     exc_ampa_nmda_ratio = None
     inh_ampa_nmda_ratio = None
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      simtime_assemb = simtime_learn

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         ids_cell_assemb = True
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime_consolidation - self.range_consolidation
         #t_start = simtime_consolidation - 3600
         #t_stop = t_start + self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.min_weight, self.has_inh_analysis, self.has_recall_metrics, self.has_weight_metrics, self.has_plots, self.has_metrics_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_recall_metrics_file, self.has_weight_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, simtime_consolidation, reptime, repdata, has_rep_presentation, self.stim))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()


 def parse_run_7_8(self):

  #prefixes_ctx = [self.file_prefix_burn, self.file_prefix_learn, self.file_prefix_consolidation, self.file_prefix_probe, self.file_prefix_cue]

  if self.run == 7:
   has_rep = False
  elif self.run == 8:
   has_rep = True
      
  for trial in self.random_trials:
   trialdir = os.path.join(self.rundir, trial)

   for brain_area in self.brain_areas:
    if brain_area == 'thl':
     
     #file_prefix_assemb = self.file_prefix_probe
     file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['ctx', 'hpc']
     nb_exc_neurons = self.nb_exc_neurons_thl
     nb_inh_neurons = self.nb_inh_neurons_thl
     has_stim_brain_area = self.has_stim_thl
     has_area_area = [self.has_ctx_thl, self.has_hpc_thl]
     prefixes = self.prefixes_thl
     full_prefixes = self.full_prefixes_thl
     exc_record_rank = self.exc_record_rank_thl
     inh_record_rank = self.inh_record_rank_thl
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_thl
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_thl
    if brain_area == 'ctx':
     
     #file_prefix_assemb = self.file_prefix_probe
     file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['hpc', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_ctx
     nb_inh_neurons = self.nb_inh_neurons_ctx
     has_stim_brain_area = self.has_stim_ctx
     has_area_area = [self.has_hpc_ctx, self.has_thl_ctx]
     prefixes = self.prefixes_ctx
     full_prefixes = self.full_prefixes_ctx
     exc_record_rank = self.exc_record_rank_ctx
     inh_record_rank = self.inh_record_rank_ctx
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_ctx
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_ctx
    elif brain_area == 'hpc':
     
     #file_prefix_assemb = self.file_prefix_probe
     file_prefix_assemb = self.file_prefix_learn
     
     other_areas = ['ctx', 'thl']
     nb_exc_neurons = self.nb_exc_neurons_hpc
     nb_inh_neurons = self.nb_inh_neurons_hpc
     has_stim_brain_area = self.has_stim_hpc
     has_area_area = [self.has_ctx_hpc, self.has_thl_hpc]
     prefixes = self.prefixes_hpc
     full_prefixes = self.full_prefixes_hpc
     exc_record_rank = self.exc_record_rank_hpc
     inh_record_rank = self.inh_record_rank_hpc
     exc_ampa_nmda_ratio = self.exc_ampa_nmda_ratio_hpc
     inh_ampa_nmda_ratio = self.inh_ampa_nmda_ratio_hpc
    plot_exc_neurons = list(np.random.choice(nb_exc_neurons, size=(self.nb_plot_neurons)))
    plot_inh_neurons = list(np.random.choice(nb_inh_neurons, size=(self.nb_plot_neurons)))

    for simtime_learn in self.simtimes_learn:
     for simtime_consolidation in self.simtimes_consolidation:
      
      #simtime_assemb = simtime_consolidation
      simtime_assemb = simtime_learn
      #if brain_area in ['ctx', 'thl']:
      # simtime_assemb = simtime_consolidation
      #elif brain_area in ['hpc']:
      # simtime_assemb = simtime_learn

      for prefix_idx in range(len(prefixes)):
       prefix = prefixes[prefix_idx]
       if brain_area == 'hpc' and self.file_prefix_hm in prefix:
        continue
       full_prefix = prefix
       if prefix == self.file_prefix_burn:
        full_prefix += '-' + str(self.simtime_burn)
        ids_cell_assemb = False
        has_stim_presentation = False
        has_rep_presentation = False
        simtime = self.simtime_burn
        t_stop = self.simtime_burn
        t_start = simtime - self.range_burn
        nb_stim = None
        stim_pat = None
        phase = self.phase_burn       
       else:
        if prefix == self.file_prefix_learn:
         prefix_end = '-' + str(self.simtime_burn)
         full_prefix += prefix_end + '-' + str(simtime_learn)
         
         #ids_cell_assemb = False
         ids_cell_assemb = True
         #if brain_area in ['hpc']:
         # ids_cell_assemb = True
         #elif brain_area in ['ctx', 'thl']:
         # ids_cell_assemb = False
         
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = simtime_learn
         t_stop = simtime_learn
         t_start = simtime - self.range_learn
         nb_stim = self.nb_stim_learn
         stim_pat = self.stim_pat_learn
         phase = self.phase_learn
        elif prefix == self.file_prefix_consolidation:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn)
         full_prefix += prefix_end + '-' + str(simtime_consolidation)
         ids_cell_assemb = False
         has_stim_presentation = False
         if has_rep:
          has_rep_presentation = True
          nb_stim = self.nb_stim_learn
          stim_pat = self.stim_pat_learn
         else:
          has_rep_presentation = False
          nb_stim = None
          stim_pat = None
         simtime = simtime_consolidation
         t_stop = simtime_consolidation
         t_start = simtime - self.range_consolidation
         phase = self.phase_consolidation
        elif prefix == self.file_prefix_probe:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_probe)
         
         #ids_cell_assemb = True
         ids_cell_assemb = False
         #if brain_area in ['hpc']:
         # ids_cell_assemb = False
         #elif brain_area in ['ctx', 'thl']:
         # ids_cell_assemb = True
         
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_probe
         t_stop = self.simtime_probe
         t_start = simtime - self.range_probe
         nb_stim = self.nb_stim_probe
         stim_pat = self.stim_pat_probe
         phase = self.phase_probe
        elif prefix == self.file_prefix_cue or prefix == self.file_prefix_cue_hm:
         prefix_end = '-' + str(self.simtime_burn) + '-' + str(simtime_learn) + '-' + str(simtime_consolidation)
         full_prefix += prefix_end + '-' + str(self.simtime_cue)
         ids_cell_assemb = False
         has_stim_presentation = True
         has_rep_presentation = False
         simtime = self.simtime_cue
         t_stop = self.simtime_cue
         t_start = simtime - self.range_cue
         nb_stim = self.nb_stim_cue
         stim_pat = self.stim_pat_cue         
         if prefix == self.file_prefix_cue_hm:
          phase = self.get_phase_hm(self.phase_cue)
         else:
          phase = self.phase_cue
        prefix += prefix_end

       analyzer_id = self.get_analyzer_id(brain_area, full_prefix)
       if analyzer_id not in self.analyzers_ids:
        full_prefixes.append(full_prefix)
       
        if has_stim_presentation:
         stimtime, stimdata = get_stim_data(trialdir, prefix, nb_stim)
        else:
         stimtime, stimdata = None, None

        if has_rep_presentation:
         reptime, repdata = get_rep_data(trialdir, prefix, self.nb_patterns)
        else:
         reptime, repdata = None, None

        self.analyzers.append(Analyzer(trial, brain_area, other_areas, phase, nb_exc_neurons, nb_inh_neurons, self.nb_neurons_stim_replay, self.nb_neuron_cons_stim_replay, has_stim_brain_area, has_area_area, has_rep, self.rundir, trialdir, prefix, full_prefix, has_stim_presentation, nb_stim, t_start, t_stop, self.integration_time_step, ids_cell_assemb, simtime_assemb, stim_pat, stimtime, stimdata, plot_exc_neurons, plot_inh_neurons, exc_record_rank, inh_record_rank, exc_ampa_nmda_ratio, inh_ampa_nmda_ratio, self.num_mpi_ranks, self.cell_assemb_method, file_prefix_assemb, self.file_prefix_hm, self.exc_min_rate, self.inh_min_rate, self.bin_size, self.nb_patterns, self.min_weight, self.has_inh_analysis, self.has_metrics, self.has_plots, self.has_rates_across_stim_plots, self.has_neuron_vi_plots, self.has_cell_assemb_plots, self.has_spike_stats_plots, self.has_activity_stats, self.has_activity_plots, self.has_spike_raster_stats, self.has_weight_plots, self.has_rf_plots, self.colors, self.zoom_range, self.has_metrics_file, self.u_rest, self.u_exc, self.u_inh, simtime, simtime_learn, simtime_consolidation, reptime, repdata, has_rep_presentation))

        self.analyzers_ids[analyzer_id] = len(self.analyzers) - 1
        print("\n\n\nParsed Analyzer:")
        self.analyzers[-1].print_identifier()

        
 def save_results(self, results_type):
  results = getattr(self, results_type + '_results')
  metrics_filename = getattr(self, results_type + '_metrics_filename')
  results.build_data_frame()
  results.save_data_frame(self.rundir, metrics_filename)
  print('\n\n\nResults Data Frame: ' + results_type)
  print(results.df)
  print('Length:', len(results.df))

  
 def plot_recall_metrics(self):

  if len(self.random_trials) == 1:
   datadir = os.path.join(self.rundir, self.random_trials[0])
  else:
   datadir = self.rundir
  if self.run == 1:
   self.plot_recall_metrics_run_1(datadir)
  else:
   self.plot_recall_metrics_run(datadir)

   
 def plot_recall_metrics_run_1(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')

  for neuron_type in neuron_types:
   
   filename = neuron_type + '_neurons-accuracy-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)))

   if self.file_prefix_cue in self.prefixes_hpc:
    filename = neuron_type + '_neurons-accuracy-phase_tests-pat_all' + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


   filename = neuron_type + '_neurons-tpr-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)))

   if self.file_prefix_cue in self.prefixes_hpc:
    filename = neuron_type + '_neurons-tpr-phase_tests-pat_all' + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


   filename = neuron_type + '_neurons-fpr-phase_all-pat_all' + extension
   if not os.path.isfile(os.path.join(datadir, filename)):
    self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)))

   if self.file_prefix_cue in self.prefixes_hpc:
    filename = neuron_type + '_neurons-fpr-phase_tests-pat_all' + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

   #'''
   for pat in range(self.nb_patterns):

    filename = neuron_type + '_neurons-accuracy-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)))

    if self.file_prefix_cue in self.prefixes_hpc:
     filename = neuron_type + '_neurons-accuracy-phase_tests-pat_' + str(pat) + extension
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


    filename = neuron_type + '_neurons-tpr-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)))

    if self.file_prefix_cue in self.prefixes_hpc:
     filename = neuron_type + '_neurons-tpr-phase_tests-pat_' + str(pat) + extension
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))


    filename = neuron_type + '_neurons-fpr-phase_all-pat_' + str(pat) + extension
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)))

    if self.file_prefix_cue in self.prefixes_hpc:
     filename = neuron_type + '_neurons-fpr-phase_tests-pat_' + str(pat) + extension
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('learn_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'training', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
   #'''


 def plot_recall_metrics_run(self, datadir):
  extension = '.svg'
  
  neuron_types = ['exc']
  if self.has_inh_analysis:
   neuron_types.append('inh')

  for neuron_type in neuron_types:
   
   for simtime_learn in self.simtimes_learn:
    suffix = self.file_prefix_learn + '-' + str(simtime_learn) + extension

    filename = neuron_type + '_neurons-accuracy-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-accuracy-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

    filename = neuron_type + '_neurons-tpr-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-tpr-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

    filename = neuron_type + '_neurons-fpr-phase_all-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
     self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)))

    filename = neuron_type + '_neurons-fpr-phase_tests-pat_all-' + suffix
    if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',('all',)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

    '''
    for pat in range(self.nb_patterns):

     filename = neuron_type + '_neurons-accuracy-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-accuracy-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'accuracy (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('accuracy',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

     filename = neuron_type + '_neurons-tpr-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-tpr-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 't.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('tpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))

     filename = neuron_type + '_neurons-fpr-phase_all-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
      self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)))

     filename = neuron_type + '_neurons-fpr-phase_tests-pat_' + str(pat) + '-' + suffix
     if not os.path.isfile(os.path.join(datadir, filename)):
       self.recall_results.plot('consolidation_time', 'score', 'region', 'phase', self.conf_int, datadir, filename, 'consolidation', 'f.p.r. (%)', ('neuron_type',(neuron_type,)), ('pattern',(str(pat),)), ('metric',('fpr',)), ('learn_time', (simtime_learn,)), ('phase', (self.phase_cue,self.get_phase_hm(self.phase_cue))))
    '''


 def compute_cluster_overlap(self, cluster1, cluster2):
  if len(cluster1) == 0:
   return 0
  return get_size_intersection(cluster1, cluster2) / float(len(cluster1))


 def compute_cluster_growth(self, cluster1, cluster2):
  if len(cluster1) == 0:
   return 0
  return len(cluster2) / float(len(cluster1))


 def get_clusters(self, area, neuron_type, simtimes, simtime, phase):
  times = []
  clusters = []
  for time in getattr(self, simtimes):
   for analyzer in self.analyzers:
    if (analyzer.brain_area == area) and (getattr(analyzer, simtime) == time) and (analyzer.phase == phase):
     times.append(time)
     clusters.append(analyzer.get_clusters(neuron_type))
  return times,clusters


 def get_cluster_overlap_growth(self, clusters):
  # overlaps: 0 idx -> pat; 0/1 -> overlap/growth
  overlaps = [[[],[]] for _ in range(self.nb_patterns + 1)]
  for time_idx in range(len(clusters) - 1):
   clusters1 = clusters[time_idx]
   clusters2 = clusters[time_idx + 1]
   for cluster1,cluster2,overlap in zip(clusters1,clusters2,overlaps):
    overlap[0].append(self.compute_cluster_overlap(cluster1, cluster2))
    overlap[1].append(self.compute_cluster_growth(cluster1, cluster2))
  return overlaps


 def plot_cluster_overlap(self, times, overlap_growth, xlabel, area, neuron_type, datadir, filename):
  fig = plt.figure()
  for cluster in range(self.nb_patterns + 1):
   overlap = overlap_growth[cluster][0]
   time = times[1:]
   if cluster == 0:
    plt.plot(time, overlap, color='black')
   else:
    plt.plot(time, overlap, color=self.colors[cluster - 1])
  sns.despine()
  plt.xlabel(xlabel + ' time (h)')
  plt.ylabel('overlap (1)')
  plt.ylim((0,1))
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s cluster overlap"%(area, neuron_type))
  plt.close(fig)

  
 def plot_cluster_growth(self, times, overlap_growth, xlabel, area, neuron_type, datadir, filename):
  fig = plt.figure()
  for cluster in range(self.nb_patterns + 1):
   growth = overlap_growth[cluster][1]
   time = times[1:]
   if cluster == 0:
    plt.plot(time, growth, color='black')
   else:
    plt.plot(time, growth, color=self.colors[cluster - 1])
  sns.despine()
  plt.xlabel(xlabel + ' time (h)')
  plt.ylabel('growth (1)')
  #plt.ylim((0,1))
  plt.savefig(os.path.join(datadir, filename), format='svg', dpi=300)
  print ("Saved %s %s cluster growth"%(area, neuron_type))
  plt.close(fig)


 def plot_cluster_overlap_growth(self, times, overlap_growth, xlabel, area, neuron_type, datadir, overlap_filename, growth_filename):
  self.plot_cluster_overlap(times, overlap_growth, xlabel, area, neuron_type, datadir, overlap_filename)
  self.plot_cluster_growth(times, overlap_growth, xlabel, area, neuron_type, datadir, growth_filename)


 def plot_clusters(self, area, neuron_type):
  has_multiple_clusters = True
  if self.run in [1]:
   simtimes = 'simtimes_learn'
   simtime = 'simtime_learn'
   phase = self.phase_learn
   xlabel = 'learning'
  elif self.run in [2,3,4]:
   simtimes = 'simtimes_consolidation'
   simtime = 'simtime_consolidation'
   phase = self.phase_consolidation
   xlabel = 'consolidation'
  else:
   has_multiple_clusters = False

  if has_multiple_clusters:
   datadir = os.path.join(self.rundir, self.random_trials[0])
   overlap_filename = area + '-' + neuron_type + '-cluster-overlap-' + phase + '.svg'
   growth_filename = area + '-' + neuron_type + '-cluster-growth-' + phase + '.svg'
   if (not os.path.isfile(os.path.join(datadir, overlap_filename))) or (not os.path.isfile(os.path.join(datadir, growth_filename))):
    print ("Plotting %s %s cluster overlap and growth..."%(area, neuron_type))
    times,clusters = self.get_clusters(area, neuron_type, simtimes, simtime, phase)
    overlap_growth = self.get_cluster_overlap_growth(clusters)
    times = np.array(times) / 60.0
    self.plot_cluster_overlap_growth(times, overlap_growth, xlabel, area, neuron_type, datadir, overlap_filename, growth_filename)


 def plot_activity_cross_correlation(self):

  neuron_types = []
  if self.has_inh_analysis:
   neuron_types.append('inh')
  neuron_types.append('exc')
  
  full_prefixes = []
  for analyzer in self.analyzers:
   if (analyzer.simtime > 0):
    full_prefixes.append(analyzer.full_prefix)
  full_prefixes = list(set(full_prefixes))

  trialdir = os.path.join(self.rundir, self.random_trials[0])
  bin_size = self.bin_size / 60
  xlabel = 'lag (min)'
  left_xlim = None
  right_xlim = None
  skip = None
  
  for area1, area2 in combinations(self.brain_areas, 2):
   for full_prefix in full_prefixes:
    for neuron_type in neuron_types:

     #'''
     analyzer1 = self.get_analyzer(area1, full_prefix)
     min_rate1 = analyzer1.get_min_rate(neuron_type)
     area_rate1 = analyzer1.get_area_rate(neuron_type)
     ca_rates1 = analyzer1.get_ca_rates(neuron_type)
     cap_rates1 = analyzer1.get_cap_rates(neuron_type)

     analyzer2 = self.get_analyzer(area2, full_prefix)
     min_rate2 = analyzer2.get_min_rate(neuron_type)
     area_rate2 = analyzer2.get_area_rate(neuron_type)
     ca_rates2 = analyzer2.get_ca_rates(neuron_type)
     cap_rates2 = analyzer2.get_cap_rates(neuron_type)
     #'''

     '''
     analyzer1 = self.get_analyzer(area1, full_prefix)
     min_rate1 = analyzer1.get_min_rate('exc')
     area_rate1 = analyzer1.get_area_rate('exc')
     ca_rates1 = analyzer1.get_ca_rates('exc')
     cap_rates1 = analyzer1.get_cap_rates('exc')

     analyzer2 = self.get_analyzer(area2, full_prefix)
     min_rate2 = analyzer2.get_min_rate('inh')
     area_rate2 = analyzer2.get_area_rate('inh')
     ca_rates2 = analyzer2.get_ca_rates('inh')
     cap_rates2 = analyzer2.get_cap_rates('inh')
     '''
     
     '''
     analyzer1 = self.get_analyzer(area1, full_prefix)
     min_rate1 = analyzer1.get_min_rate('inh')
     area_rate1 = analyzer1.get_area_rate('inh')
     ca_rates1 = analyzer1.get_ca_rates('inh')
     cap_rates1 = analyzer1.get_cap_rates('inh')

     analyzer2 = self.get_analyzer(area2, full_prefix)
     min_rate2 = analyzer2.get_min_rate('exc')
     area_rate2 = analyzer2.get_area_rate('exc')
     ca_rates2 = analyzer2.get_ca_rates('exc')
     cap_rates2 = analyzer2.get_cap_rates('exc')
     '''

     '''
     other_analyzer1 = self.get_analyzer(area1, full_prefix)
     other_min_rate1 = analyzer1.get_min_rate(neuron_type)
     other_area_rate1 = analyzer1.get_area_rate(neuron_type)
     other_ca_rates1 = analyzer1.get_ca_rates(neuron_type)
     other_cap_rates1 = analyzer1.get_cap_rates(neuron_type)

     other_analyzer2 = self.get_analyzer(area2, full_prefix)
     other_min_rate2 = analyzer2.get_min_rate(neuron_type)
     other_area_rate2 = analyzer2.get_area_rate(neuron_type)
     other_ca_rates2 = analyzer2.get_ca_rates(neuron_type)
     other_cap_rates2 = analyzer2.get_cap_rates(neuron_type)
     '''

     assert analyzer1.t_start == analyzer2.t_start and analyzer1.t_stop == analyzer2.t_stop,"Inconsistent t_start and t_stop between brain regions for activity cross-correlation"

     t_start = analyzer1.t_start
     t_stop = analyzer1.t_stop

     #print('Activity Cross-Correlation - Parser')
     #print(area1, area2)
     #print(type(area_rate1), type(area_rate2))
     #print(type(ca_rates1), type(ca_rates2))
     
     #'''
     filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_area-' + area2 + '_area-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
     plot_activity_cross_correlation('area', 'area', 'gray', self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, area_rate1[0], area_rate2[0], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)
     #'''
     
     if type(ca_rates1) != type(None) and type(ca_rates2) != type(None):
      #for stim1, stim2 in product(range(self.nb_patterns), range(self.nb_patterns)):
      #'''
      for stim1, stim2 in combinations(range(self.nb_patterns), 2):

       filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_ca_' + str(stim1) + '-' + area2 + '_ca_' + str(stim2) + '-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
       plot_activity_cross_correlation('ca_' + str(stim1), 'ca_' + str(stim2), '#FF00FF', self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, ca_rates1[stim1], ca_rates2[stim2], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

       filename = 'activity_cross_correlation-' + neuron_type + '-' + area2 + '_ca_' + str(stim1) + '-' + area1 + '_ca_' + str(stim2) + '-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
       plot_activity_cross_correlation('ca_' + str(stim1), 'ca_' + str(stim2), '#FF00FF', self.cell_assemb_method, trialdir, filename, area2, area1, neuron_type, ca_rates2[stim1], ca_rates1[stim2], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip) 
      #'''
      
      #'''
      for stim in range(self.nb_patterns):
       filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_ca_' + str(stim) + '-' + area2 + '_ca_' + str(stim) + '-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
       plot_activity_cross_correlation('ca_' + str(stim), 'ca_' + str(stim), self.colors[stim], self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, ca_rates1[stim], ca_rates2[stim], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)
      #'''

     #'''
     if type(cap_rates1) != type(None) and type(cap_rates2) != type(None):
      filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_no_stim-' + area2 + '_no_stim-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
      plot_activity_cross_correlation('no_stim', 'no_stim', 'black', self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, cap_rates1[0], cap_rates2[0], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

      if type(ca_rates1) != type(None) and type(ca_rates2) != type(None):
       for stim in range(self.nb_patterns):
        filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_ca_' + str(stim) + '-' + area2 + '_no_stim-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
        plot_activity_cross_correlation('ca_' + str(stim), 'no_stim', '#FF00FF', self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, ca_rates1[stim], cap_rates2[0], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)

        filename = 'activity_cross_correlation-' + neuron_type + '-' + area1 + '_no_stim-' + area2 + '_ca_' + str(stim) + '-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
        plot_activity_cross_correlation('no_stim', 'ca_' + str(stim), '#FF00FF', self.cell_assemb_method, trialdir, filename, area1, area2, neuron_type, cap_rates1[0], ca_rates2[stim], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)
     #'''

  if neuron_types[0] == 'inh' and neuron_types[1] == 'exc':
   for area1, area2 in combinations(self.brain_areas, 2):
    for full_prefix in full_prefixes:
    
     analyzer1 = self.get_analyzer(area1, full_prefix)
     min_rate1 = analyzer1.get_min_rate('inh')
     area_rate1 = analyzer1.get_area_rate('inh')
     ca_rates1 = analyzer1.get_ca_rates('inh')
     cap_rates1 = analyzer1.get_cap_rates('inh')
      
     analyzer2 = self.get_analyzer(area2, full_prefix)
     min_rate2 = analyzer2.get_min_rate('exc')
     area_rate2 = analyzer2.get_area_rate('exc')
     ca_rates2 = analyzer2.get_ca_rates('exc')
     cap_rates2 = analyzer2.get_cap_rates('exc')

     assert analyzer1.t_start == analyzer2.t_start and analyzer1.t_stop == analyzer2.t_stop,"Inconsistent t_start and t_stop between brain regions for activity cross-correlation"

     t_start = analyzer1.t_start
     t_stop = analyzer1.t_stop
     
     if type(ca_rates1) != type(None) and type(ca_rates2) != type(None):
      for stim in range(self.nb_patterns):
       filename = 'activity_cross_correlation-' + 'exc-inh-' + area2 + '_ca_' + str(stim) + '-' + area1 + '_ca_' + str(stim) + '-' + self.cell_assemb_method + '-' + full_prefix + '-t_start-' + str(t_start) + '-t_stop-' + str(t_stop)
       plot_activity_cross_correlation('ca_' + str(stim), 'ca_' + str(stim), self.colors[stim], self.cell_assemb_method, trialdir, filename, area2, area1, neuron_type, ca_rates2[stim], ca_rates1[stim], bin_size=bin_size, xlabel=xlabel, left_xlim=left_xlim, right_xlim=right_xlim, skip=skip)




 def plot_dynamics(self):
  print ("\n\nRunning Parser dynamic plots...")
  
  if self.has_cell_assemb_plots:
   for area in self.brain_areas:
    self.plot_clusters(area, 'exc')
    if self.has_inh_analysis:
     self.plot_clusters(area, 'inh')

  if self.has_activity_plots:
   self.plot_activity_cross_correlation()
   


 def plot_area_area_weights(self, compute_weight_metrics):
  for analyzer in self.analyzers:
   if (analyzer.simtime > 0):
    print ("\nRunning Parser Cross-Area Weight Plots...")
    analyzer.print_identifier()
    for other_area in analyzer.other_areas:
     other_area_ana = self.get_analyzer(other_area, analyzer.full_prefix)
     if type(other_area_ana) != type(None):

      if compute_weight_metrics:
       results = self.weight_results
      else:
       results = None
       
      prefixes = [analyzer.full_prefix]
      if analyzer.phase in ['burn-in']:
       prefixes.append(analyzer.prefix)

      for prefix in prefixes:
       print(prefix + ':')
       analyzer.plot_weights(prefix,
                             has_rec_plots=False,
                             has_stim_brain_area_plots=False,
                             has_area_area_plots=True,
                             other_area=other_area,
                             other_area_ana=other_area_ana,
                             compute_weight_metrics=compute_weight_metrics,
                             results=results)


 def plot_cluster_incoming_weights(self):
  '''
  Plot sum of incoming weights to neurons of every excitatory cluster by projection and in total.
  '''
  
  for analyzer in self.analyzers:
   if analyzer.simtime > 0:
    print ("\nRunning Parser Incoming Weights Plots...")
    analyzer.print_identifier()

    other_area_anas = []
    for other_area in analyzer.other_areas:
     other_area_anas.append(self.get_analyzer(other_area, analyzer.full_prefix))
      
    #if self.file_prefix_hm not in analyzer.full_prefix:
    # other_area_ana = self.get_analyzer(analyzer.other_areas, analyzer.full_prefix)
    #else:
    # other_area_ana = None
    
    prefixes = [analyzer.full_prefix]
    if analyzer.phase == 'burn-in':
     prefixes.append(analyzer.prefix)
    for prefix in prefixes:
     print(prefix + ':')
     analyzer.plot_cluster_incoming_weights(prefix, other_area_anas)


 def plot_cross_time_ca_incoming_weights(self):
  '''
  Plot and KS-test the CDF of the sum of weights to engram cells for each plastic connection at 
  every (learning_time, consolidation_time) pair.
  '''

  print("\n\n\nPlotting weight CDF for every pair of learning and consolidation timepoints")
  
  learn_analyzers = []
  con_analyzers = []
  for analyzer in self.analyzers:
   if analyzer.simtime > 0:
    if analyzer.phase == self.phase_learn:
     learn_analyzers.append(analyzer)
    if analyzer.phase == self.phase_consolidation:
     con_analyzers.append(analyzer)

  for  brain_area in self.brain_areas:
   for learn_analyzer in learn_analyzers:
    if learn_analyzer.brain_area == brain_area:
     
     learn_inh_in_weights = learn_analyzer.get_ca_incoming_weights(brain_area, brain_area, 'ie', learn_analyzer.full_prefix, learn_analyzer)
     learn_exc_in_weights = learn_analyzer.get_ca_incoming_weights(brain_area, brain_area, 'ee', learn_analyzer.full_prefix, learn_analyzer)
     if learn_analyzer.has_stim_brain_area:
      learn_stim_in_weights = learn_analyzer.get_ca_incoming_weights('stim', brain_area, 'ee', learn_analyzer.full_prefix)
     
     for con_analyzer in con_analyzers:
      if con_analyzer.brain_area == brain_area:
       
       con_inh_in_weights = con_analyzer.get_ca_incoming_weights(brain_area, brain_area, 'ie', con_analyzer.full_prefix, con_analyzer)
       filename = 'incoming_weights_cdf-' + brain_area + '-ie-all_cas-' + self.phase_learn + '_to_' + self.phase_consolidation + '-' + con_analyzer.full_prefix + '.svg'
       plot_comp_weight_cdf(brain_area, 'ie', learn_inh_in_weights, con_inh_in_weights, '$t_{consolidation}=0h$', '$t_{consolidation}=%dh$'%(int(con_analyzer.simtime)), 'black', 'gray', 'inhibitory weights', learn_analyzer.trialdir, filename)
       _, ks_pvalue = kstest(learn_inh_in_weights.flatten(), con_inh_in_weights.flatten(), alternative='two-sided', mode='auto')
       self.weight_results.add_result(self.random_trials[0], brain_area, 'ie', learn_analyzer.simtime, con_analyzer.simtime, learn_analyzer.phase + '_' + con_analyzer.phase, 'all', 'cross_time_ca_in_weight_ks_pvalue', ks_pvalue)
       
       con_exc_in_weights = con_analyzer.get_ca_incoming_weights(brain_area, brain_area, 'ee', con_analyzer.full_prefix, con_analyzer)
       filename = 'incoming_weights_cdf-' + brain_area + '-ee-all_cas-' + self.phase_learn + '_to_' + self.phase_consolidation + '-' + con_analyzer.full_prefix + '.svg'
       plot_comp_weight_cdf(brain_area, 'ee', learn_exc_in_weights, con_exc_in_weights, '$t_{consolidation}=0h$', '$t_{consolidation}=%dh$'%(int(con_analyzer.simtime)), 'black', 'gray', 'recurrent excitatory weights', learn_analyzer.trialdir, filename)
       _, ks_pvalue = kstest(learn_exc_in_weights.flatten(), con_exc_in_weights.flatten(), alternative='two-sided', mode='auto')
       self.weight_results.add_result(self.random_trials[0], brain_area, 'ee', learn_analyzer.simtime, con_analyzer.simtime, learn_analyzer.phase + '_' + con_analyzer.phase, 'all', 'cross_time_ca_in_weight_ks_pvalue', ks_pvalue)

       if con_analyzer.has_stim_brain_area:
        con_stim_in_weights = con_analyzer.get_ca_incoming_weights('stim', brain_area, 'ee', con_analyzer.full_prefix)
        filename = 'incoming_weights_cdf-stim_' + brain_area + '-ee-all_cas-' + self.phase_learn + '_to_' + self.phase_consolidation + '-' + con_analyzer.full_prefix + '.svg'
        plot_comp_weight_cdf('stim_' + brain_area, 'ee', learn_stim_in_weights, con_stim_in_weights, '$t_{consolidation}=0h$', '$t_{consolidation}=%dh$'%(int(con_analyzer.simtime)), 'black', 'gray', 'stimulus weights', learn_analyzer.trialdir, filename)
        _, ks_pvalue = kstest(learn_stim_in_weights.flatten(), con_stim_in_weights.flatten(), alternative='two-sided', mode='auto')
        self.weight_results.add_result(self.random_trials[0], 'stim_' + brain_area, 'ee', learn_analyzer.simtime, con_analyzer.simtime, learn_analyzer.phase + '_' + con_analyzer.phase, 'all', 'cross_time_ca_in_weight_ks_pvalue', ks_pvalue)

       for other_area in learn_analyzer.other_areas:
        learn_other_area_ana = self.get_analyzer(other_area, learn_analyzer.full_prefix)
        con_other_area_ana = self.get_analyzer(other_area, con_analyzer.full_prefix)
        if type(learn_other_area_ana) != type(None) and type(con_other_area_ana) != type(None):
         other_area_idx = learn_analyzer.other_areas.index(other_area)
         has_area_area = learn_analyzer.has_area_area[other_area_idx]
         if has_area_area:
          learn_other_area_in_weights = learn_analyzer.get_ca_incoming_weights(other_area, brain_area, 'ee', learn_analyzer.full_prefix, learn_other_area_ana)
          con_other_area_in_weights = con_analyzer.get_ca_incoming_weights(other_area, brain_area, 'ee', con_analyzer.full_prefix, con_other_area_ana)
          filename = 'incoming_weights_cdf-' + other_area + '_' + brain_area + '-ee-all_cas-' + self.phase_learn + '_to_' + self.phase_consolidation + '-' + con_analyzer.full_prefix + '.svg'
          plot_comp_weight_cdf(other_area + '_' + brain_area, 'ee', learn_other_area_in_weights, con_other_area_in_weights, '$t_{consolidation}=0h$', '$t_{consolidation}=%dh$'%(int(con_analyzer.simtime)), 'black', 'gray', 'feedforward weights', learn_analyzer.trialdir, filename)
          _, ks_pvalue = kstest(learn_other_area_in_weights.flatten(), con_other_area_in_weights.flatten(), alternative='two-sided', mode='auto')
          self.weight_results.add_result(self.random_trials[0], other_area + '_' + brain_area, 'ee', learn_analyzer.simtime, con_analyzer.simtime, learn_analyzer.phase + '_' + con_analyzer.phase, 'all', 'cross_time_ca_in_weight_ks_pvalue', ks_pvalue)
         
        

  # Cumulative weight distribution over time
  '''
  print('self.full_prefixes_thl',self.full_prefixes_thl)
  print('self.full_prefixes_ctx',self.full_prefixes_ctx)
  print('self.full_prefixes_hpc',self.full_prefixes_hpc)
  '''

  # Inhibitory Recurrent Weights
  #
  
  '''
  learn_prefix = self.full_prefixes_ctx[1]
  learn_analyzer = self.get_analyzer('ctx', learn_prefix)
  learn_inh_in_weights = learn_analyzer.get_ca_incoming_weights('ctx', learn_analyzer.brain_area, 'ie', learn_prefix, learn_analyzer)

  con_prefix = self.full_prefixes_ctx[2]
  con_analyzer = self.get_analyzer('ctx', con_prefix)
  con_inh_in_weights = con_analyzer.get_ca_incoming_weights('ctx', con_analyzer.brain_area, 'ie', con_prefix, con_analyzer)

  #print('learn_in', np.average(learn_inh_in_weights))
  #print('con_in', np.average(con_inh_in_weights))

  filename = 'rfl-1800-ctx-incoming_weights-cdf-inh_ctx-all_cas-from_rfl_to_rfc-120-1800-43200.svg'
  plot_comp_weight_cdf('ctx', 'inh_ctx', learn_inh_in_weights, con_inh_in_weights, self.colors, learn_analyzer.trialdir, filename)
  '''

  '''
  for area in ['ctx', 'thl', 'hpc']:
   learn_prefix = self.full_prefixes_ctx[1]
   learn_analyzer = self.get_analyzer(area, learn_prefix)
   learn_inh_in_weights = learn_analyzer.get_ca_incoming_weights(area, learn_analyzer.brain_area, 'ie', learn_prefix, learn_analyzer)

   con_prefix = self.full_prefixes_ctx[2]
   con_analyzer = self.get_analyzer(area, con_prefix)
   con_inh_in_weights = con_analyzer.get_ca_incoming_weights(area, con_analyzer.brain_area, 'ie', con_prefix, con_analyzer)

   print('learn_in', np.average(learn_inh_in_weights))
   print('con_in', np.average(con_inh_in_weights))

   filename = 'rfl-1800-' + area + '-incoming_weights-cdf-inh_' + area + '-all_cas-from_rfl_to_rfc-120-1800-43200.svg'
   plot_comp_weight_cdf(area, 'ie', learn_inh_in_weights, con_inh_in_weights, self.colors, learn_analyzer.trialdir, filename)
  '''
  
  '''
  learn_prefix = self.full_prefixes_hpc[0]
  learn_analyzer = self.get_analyzer('hpc', learn_prefix)
  learn_inh_in_weights = learn_analyzer.get_ca_incoming_weights('stim', learn_analyzer.brain_area, 'ee', learn_prefix)

  con_prefix = self.full_prefixes_hpc[2]
  con_analyzer = self.get_analyzer('hpc', con_prefix)
  con_inh_in_weights = con_analyzer.get_ca_incoming_weights('stim', con_analyzer.brain_area, 'ee', con_prefix)

  print('learn_in', np.average(learn_inh_in_weights))
  print('con_in', np.average(con_inh_in_weights))

  filename = 'rfl-2700-hpc-incoming_weights-cdf-inh_hpc-all_cas-from_rfl_to_rfc-120-2700-43200'
  plot_comp_weight_cdf('hpc', 'inh_hpc', learn_inh_in_weights, con_inh_in_weights, self.colors, learn_analyzer.trialdir, filename)
  '''

  # STIM feedforward weights
  '''
  learn_prefix = self.full_prefixes_ctx[0]
  learn_analyzer = self.get_analyzer('ctx', learn_prefix)
  learn_stim_in_weights = learn_analyzer.get_ca_incoming_weights('stim', learn_analyzer.brain_area, 'ee', learn_prefix)

  con_prefix = self.full_prefixes_ctx[2]
  con_analyzer = self.get_analyzer('ctx', con_prefix)
  con_stim_in_weights = con_analyzer.get_ca_incoming_weights('stim', con_analyzer.brain_area, 'ee', con_prefix)

  print('learn_in', np.average(learn_stim_in_weights))
  print('con_in', np.average(con_stim_in_weights))

  filename = 'rfl-2700-ctx-incoming_weights-cdf-stim_ctx-all_cas-from_rfl_to_rfc-120-2700-43200'
  plot_comp_weight_cdf('ctx', 'stim_ctx', learn_stim_in_weights, con_stim_in_weights, self.colors, learn_analyzer.trialdir, filename)

  learn_prefix = self.full_prefixes_hpc[1]
  learn_analyzer = self.get_analyzer('hpc', learn_prefix)
  learn_stim_in_weights = learn_analyzer.get_ca_incoming_weights('stim', learn_analyzer.brain_area, 'ee', learn_prefix)

  con_prefix = self.full_prefixes_hpc[2]
  con_analyzer = self.get_analyzer('hpc', con_prefix)
  con_stim_in_weights = con_analyzer.get_ca_incoming_weights('stim', con_analyzer.brain_area, 'ee', con_prefix)

  print('learn_in', np.average(learn_stim_in_weights))
  print('con_in', np.average(con_stim_in_weights))

  filename = 'rfl-1800-hpc-incoming_weights-cdf-stim_hpc-all_cas-from_rfl_to_rfc-120-1800-86400.svg'
  plot_comp_weight_cdf('hpc', 'stim_hpc', learn_stim_in_weights, con_stim_in_weights, self.colors, learn_analyzer.trialdir, filename)
  '''

  # THL->CTX feedforward weights
  '''
  learn_prefix = 'rfl-120-1800'
  post_learn_analyzer = self.get_analyzer('ctx', learn_prefix)
  pre_learn_analyzer = self.get_analyzer('thl', learn_prefix)
  learn_in_weights = post_learn_analyzer.get_ca_incoming_weights('thl', 'ctx', 'ee', learn_prefix, pre_learn_analyzer)

  con_prefix = 'rfc-120-1800-86400'
  post_con_analyzer = self.get_analyzer('ctx', con_prefix)
  pre_con_analyzer = self.get_analyzer('thl', con_prefix)
  con_in_weights = post_con_analyzer.get_ca_incoming_weights('thl', 'ctx', 'ee', con_prefix, pre_con_analyzer)

  print('learn_in', np.average(learn_in_weights))
  print('con_in', np.average(con_in_weights))

  filename = 'rfl-1800-thl_ctx-incoming_weights-cdf-thl_ctx-all_cas-from_rfl_to_rfc-120-1800-86400.svg'
  plot_comp_weight_cdf('ctx', 'thl_ctx', learn_in_weights, con_in_weights, self.colors, post_learn_analyzer.trialdir, filename)
  '''


 def plot_weights(self, compute_weight_metrics=False):
  print ("\n\nRunning Parser Weight Plots...")
  self.plot_area_area_weights(compute_weight_metrics)
  
  if compute_weight_metrics and self.run != 1:
   self.plot_cross_time_ca_incoming_weights()
  
  if not compute_weight_metrics:
   self.plot_cluster_incoming_weights()


 def merge_metrics(self, metrics_type):
  metrics_filename = getattr(self, metrics_type + '_metrics_filename')
  print ("Merging %s metrics of individual trials..."%(metrics_type))
  frames = []
  for metrics in ['metrics-'+ metrics_type + '-' + str(i) + '.csv' for i in self.random_trials]:
   print (metrics)
   frames.append(pd.read_csv(os.path.join(self.rundir, metrics)))
  metrics_all = pd.concat(frames, ignore_index=True)
  metrics_all.to_csv(os.path.join(self.rundir, metrics_filename))


 def analyze(self):
  '''
  List of saved files:
  - metrics (*.metrics)
  '''

  if len(self.random_trials) == 1:
   for analyzer in self.analyzers:
    analyzer.analyze(self.recall_results, self.weight_results)

   if self.has_weight_metrics and (not self.has_weight_metrics_file):
    self.plot_weights(compute_weight_metrics=True)
    
   if self.has_plots:
    self.plot_dynamics()
    if self.has_weight_plots:
     self.plot_weights()

  
  if self.has_recall_metrics:
   if self.has_recall_metrics_file:
    self.recall_results.load_df(self.rundir, self.recall_metrics_filename)
   else:
    if len(self.random_trials) == 1:
     self.save_results('recall')
     self.recall_results.load_df(self.rundir, self.recall_metrics_filename)
    else:
     self.merge_metrics('recall')
     self.recall_results.load_df(self.rundir, self.recall_metrics_filename)
     
  if self.has_weight_metrics:
   if self.has_weight_metrics_file:
    self.weight_results.load_df(self.rundir, self.weight_metrics_filename)
   else:
    if len(self.random_trials) == 1:
     self.save_results('weight')
     self.weight_results.load_df(self.rundir, self.weight_metrics_filename)
    else:
     self.merge_metrics('weight')
     self.weight_results.load_df(self.rundir, self.weight_metrics_filename)

  if self.has_recall_metrics and self.has_metrics_plots:
   self.plot_recall_metrics()

