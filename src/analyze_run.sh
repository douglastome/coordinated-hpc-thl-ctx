#!/bin/bash
# Command-line arguments:
# 1: run type
# 2: random trials (comma-separated list e.g. 0,1,2,3)
# 3: brain areas (comma-separated list e.g. thl,ctx,hpc)
# 4: thl prefixes (comma-separated list e.g. $FILE_PREFIX_LEARN,$FILE_PREFIX_CUE)
# 5: ctx prefixes (comma-separated list e.g. $FILE_PREFIX_LEARN,$FILE_PREFIX_CUE)
# 6: hpc prefixes (comma-separated list e.g. $FILE_PREFIX_LEARN,$FILE_PREFIX_CUE)
# 7: rdt prefixes (comma-separated list e.g. $FILE_PREFIX_LEARN,$FILE_PREFIX_CUE)
# 8: flag to compute and save recall metrics (i.e. true or false)
# 9: flag to compute and save weight metrics (i.e. true or false)
# 10: flag to enable plotting (i.e. true or false)
# 11: run id (e.g., run-001)

# parse command-line arguments
RUN_TYPE=$1
RANDOM_TRIALS=$2
BRAIN_AREAS=[$3]
PREFIXES_THL=[$4]
PREFIXES_CTX=[$5]
PREFIXES_HPC=[$6]
PREFIXES_RDT=[$7]
HAS_RECALL_METRICS=$8
HAS_WEIGHT_METRICS=$9
HAS_PLOTS=${10}
RUN_ID=${11}

# output directory
OUT_DIR="$HOME/projects/systems-consolidation/simulations/sim_rc_p11/$RUN_ID"

# source directory
SOURCE="src"
DIR=$OUT_DIR/$SOURCE

. $DIR/globalvars.sh

# analysis flags
HAS_INH_ANALYSIS="true"
HAS_ACTIVITY_STATS="false"
HAS_SPIKE_RASTER_STATS="false"
HAS_MERGE="false"

# plotting flags
HAS_METRICS_PLOTS="true"
HAS_CELL_ASSEMB_PLOTS="false"
HAS_RATES_ACROSS_STIM_PLOTS="false"
HAS_ACTIVITY_PLOTS="false"
HAS_SPIKE_STATS_PLOTS="false"
HAS_WEIGHT_PLOTS="false"
HAS_RF_PLOTS="false"
HAS_NEURON_VI_PLOTS="false"

# architecture
HAS_STIM_THL="true"
HAS_STIM_CTX="false"
HAS_STIM_HPC="true"
HAS_STIM_RDT="false"

HAS_THL_CTX="true"
HAS_CTX_THL="false"

HAS_THL_HPC="false"
HAS_HPC_THL="true"

HAS_CTX_HPC="false"
HAS_HPC_CTX="false"

HAS_THL_RDT="false"
HAS_RDT_THL="false"

HAS_CTX_RDT="false"
HAS_RDT_CTX="false"

HAS_HPC_RDT="false"
HAS_RDT_HPC="false"

HAS_HPC_REP="false"

# stimulation
NB_PATTERNS=4

NB_STIM_LEARN=4
STIM_PAT_LEARN="(0,0),(1,1),(2,2),(3,3)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_LEARN="(0,0)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_LEARN="(0,0),(1,1),(2,2)" # comma-separated (stimulus,pattern) pairs

NB_STIM_PROBE=4
STIM_PAT_PROBE="(0,0),(1,1),(2,2),(3,3)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_PROBE="(0,0)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_PROBE="(0,0),(1,1),(2,2)" # comma-separated (stimulus,pattern) pairs

NB_STIM_CUE=4
STIM_PAT_CUE="(0,0),(1,1),(2,2),(3,3)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_CUE="(0,0),(1,0),(2,0),(3,0),(4,1),(5,1),(6,1),(7,1),(8,2),(9,2),(10,2),(11,2),(12,3),(13,3),(14,3),(15,3)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_CUE="(0,0),(1,0),(2,0),(3,0)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_CUE="(0,0)" # comma-separated (stimulus,pattern) pairs
#STIM_PAT_CUE="(0,0),(1,1),(2,2)" # comma-separated (stimulus,pattern) pairs

# analysis
RANGE_BURN=120
RANGE_LEARN=300
RANGE_CONSOLIDATION=1800
RANGE_PROBE=300
RANGE_CUE=60
CELL_ASSEMB_METHOD="rate" #"rate" or "nfm"
EXC_MIN_RATE=10
INH_MIN_RATE=10
MIN_WEIGHT=0.5
BIN_SIZE="0.01"
CONF_INT=90
SIMTIME_LEARN_MERGE=1800

# plotting
ZOOM_RANGE="1.0"
PHASE_BURN="burn-in"
PHASE_LEARN="learn"
PHASE_CONSOLIDATION="consolidation"
PHASE_PROBE="probe"
PHASE_CUE="test"
NB_PLOT_NEURONS=256
#COLORS="[sandybrown,darkseagreen,royalblue,thistle]"
COLORS="[(0.4,0.0,0.0),(0.0,0.0,0.4),(0.0,0.4,0.0),(0.4,0.0,0.4)]"
#COLORS="[(0.4,0.0,0.0),(0.0,0.4,0.0),(0.4,0.0,0.4)]"

case $RUN_TYPE in
    1)
	# if simtime_learn=0 is valid:
	# uncomment the following:
	#SIMTIMES_LEARN=0
	#START_IDX=0
	# else, uncomment the following:
	SIMTIMES_LEARN=${T_STOP_LEARN[0]}
	START_IDX=1

	for ((idx=$START_IDX; idx<"${#T_STOP_LEARN[@]}"; idx++))
	do
	    SIMTIMES_LEARN="$SIMTIMES_LEARN,${T_STOP_LEARN[$idx]}"
	done
	#SIMTIMES_LEARN=1800
	;;
    2 | 3 | 4 | 5 | 6 | 7 | 8)
	if [ $HAS_MERGE = true  ]
	then
	    SIMTIMES_LEARN=$SIMTIME_LEARN_MERGE
	else
	    SIMTIMES_LEARN=${T_STOP_LEARN_CON[$RANDOM_TRIALS]}
	fi
	;;
    *)
	echo RUN_TYPE $RUN_TYPE : UNKOWN
	;;
esac

SIMTIMES_CONSOLIDATION=0
for ((idx=0; idx<"${#T_STOP_CONSOLIDATION[@]}"; idx++))
do
    SIMTIMES_CONSOLIDATION="$SIMTIMES_CONSOLIDATION,${T_STOP_CONSOLIDATION[$idx]}"
done
#SIMTIMES_CONSOLIDATION=86400
#SIMTIMES_CONSOLIDATION="$SIMTIME_CONSOLIDATION_8,$SIMTIME_CONSOLIDATION_7,$SIMTIME_CONSOLIDATION_6,$SIMTIME_CONSOLIDATION_5,$SIMTIME_CONSOLIDATION_4,$SIMTIME_CONSOLIDATION_3,$SIMTIME_CONSOLIDATION_2,$SIMTIME_CONSOLIDATION_1,0"

# Check command-line arguments:
echo RUN_TYPE $RUN_TYPE
echo RANDOM_TRIALS $RANDOM_TRIALS
echo BRAIN_AREAS $BRAIN_AREAS
echo PREFIXES_THL $PREFIXES_THL
echo PREFIXES_CTX $PREFIXES_CTX
echo PREFIXES_HPC $PREFIXES_HPC
echo PREFIXES_RDT $PREFIXES_RDT
echo HAS_PLOTS $HAS_PLOTS
echo HAS_MERGE $HAS_MERGE
echo SIMTIME_LEARN_MERGE $SIMTIME_LEARN_MERGE
echo SIMTIMES_LEARN $SIMTIMES_LEARN

# analyze run
~/venv_sim/bin/python $DIR/analyze_run.py --run $RUN_TYPE --has_inh_analysis $HAS_INH_ANALYSIS --has_recall_metrics $HAS_RECALL_METRICS --has_weight_metrics $HAS_WEIGHT_METRICS --has_plots $HAS_PLOTS --has_neuron_vi_plots $HAS_NEURON_VI_PLOTS --has_cell_assemb_plots $HAS_CELL_ASSEMB_PLOTS --has_activity_stats $HAS_ACTIVITY_STATS --has_activity_plots $HAS_ACTIVITY_PLOTS --has_spike_raster_stats $HAS_SPIKE_RASTER_STATS --has_spike_stats_plots $HAS_SPIKE_STATS_PLOTS --has_metrics_plots $HAS_METRICS_PLOTS --has_rates_across_stim_plots $HAS_RATES_ACROSS_STIM_PLOTS --has_weight_plots $HAS_WEIGHT_PLOTS --has_rf_plots $HAS_RF_PLOTS --conf_int $CONF_INT --n_ranks $NP --time_step $INTEGRATION_TIME_STEP --rundir $OUT_DIR --trials $RANDOM_TRIALS --brain_areas $BRAIN_AREAS --prefixes_thl $PREFIXES_THL --prefixes_ctx $PREFIXES_CTX --prefixes_hpc $PREFIXES_HPC --prefixes_rdt $PREFIXES_RDT --file_prefix_burn $FILE_PREFIX_BURN --file_prefix_learn $FILE_PREFIX_LEARN --file_prefix_consolidation $FILE_PREFIX_CONSOLIDATION --file_prefix_probe $FILE_PREFIX_PROBE --file_prefix_cue $FILE_PREFIX_CUE --file_prefix_hm $FILE_PREFIX_HM --file_prefix_burn_hm $FILE_PREFIX_BURN_HM --file_prefix_learn_hm $FILE_PREFIX_LEARN_HM --file_prefix_consolidation_hm $FILE_PREFIX_CONSOLIDATION_HM --file_prefix_probe_hm $FILE_PREFIX_PROBE_HM --file_prefix_cue_hm $FILE_PREFIX_CUE_HM --phase_burn $PHASE_BURN --phase_learn $PHASE_LEARN --phase_consolidation $PHASE_CONSOLIDATION --phase_probe $PHASE_PROBE --phase_cue $PHASE_CUE --n_stim_learn $NB_STIM_LEARN --n_stim_probe $NB_STIM_PROBE --n_stim_cue $NB_STIM_CUE --exc_size_thl $EXC_SIZE_THL --exc_inh_thl $EXC_INH_THL --exc_size_ctx $EXC_SIZE_CTX --exc_inh_ctx $EXC_INH_CTX --exc_size_hpc $EXC_SIZE_HPC --exc_inh_hpc $EXC_INH_HPC --exc_size_rdt $EXC_SIZE_RDT --exc_inh_rdt $EXC_INH_RDT --stim_replay $NB_NEURONS_STIM_REP --cons_stim_replay $NB_NEURON_CONS_STIM_REP --n_patterns $NB_PATTERNS --simtime_burn $SIMTIME_BURN --simtimes_learn $SIMTIMES_LEARN --simtimes_consolidation $SIMTIMES_CONSOLIDATION --simtime_probe $SIMTIME_PROBE --simtime_cue $SIMTIME_CUE --range_burn $RANGE_BURN --range_learn $RANGE_LEARN --range_consolidation $RANGE_CONSOLIDATION --range_probe $RANGE_PROBE --range_cue $RANGE_CUE --zoom_range $ZOOM_RANGE --n_plot_neurons $NB_PLOT_NEURONS --colors $COLORS --bin $BIN_SIZE --has_stim_hpc $HAS_STIM_HPC --has_stim_ctx $HAS_STIM_CTX --has_stim_thl $HAS_STIM_THL --has_stim_rdt $HAS_STIM_RDT --has_thl_ctx $HAS_THL_CTX --has_ctx_thl $HAS_CTX_THL --has_thl_hpc $HAS_THL_HPC --has_hpc_thl $HAS_HPC_THL --has_ctx_hpc $HAS_CTX_HPC --has_hpc_ctx $HAS_HPC_CTX --has_thl_rdt $HAS_THL_RDT --has_ctx_rdt $HAS_CTX_RDT --has_hpc_rdt $HAS_HPC_RDT --has_rdt_thl $HAS_RDT_THL --has_rdt_ctx $HAS_RDT_CTX --has_rdt_hpc $HAS_RDT_HPC --has_hpc_rep $HAS_HPC_REP --ids_cell_assemb true --cell_assemb_method $CELL_ASSEMB_METHOD --exc_min_rate $EXC_MIN_RATE --inh_min_rate $INH_MIN_RATE --min_weight $MIN_WEIGHT --stim_pat_learn $STIM_PAT_LEARN --stim_pat_probe $STIM_PAT_PROBE --stim_pat_cue $STIM_PAT_CUE --exc_record_rank_thl $EXC_RECORD_RANK_THL --inh_record_rank_thl $INH_RECORD_RANK_THL --exc_ampa_nmda_ratio_thl $AMPA_NMDA_E_THL --inh_ampa_nmda_ratio_thl $AMPA_NMDA_I_THL --exc_record_rank_ctx $EXC_RECORD_RANK_CTX --inh_record_rank_ctx $INH_RECORD_RANK_CTX --exc_ampa_nmda_ratio_ctx $AMPA_NMDA_E_CTX --inh_ampa_nmda_ratio_ctx $AMPA_NMDA_I_CTX --exc_record_rank_hpc $EXC_RECORD_RANK_HPC --inh_record_rank_hpc $INH_RECORD_RANK_HPC --exc_ampa_nmda_ratio_hpc $AMPA_NMDA_E_HPC --inh_ampa_nmda_ratio_hpc $AMPA_NMDA_I_HPC --u_rest $U_REST --u_exc $U_EXC --u_inh $U_INH --stim $STIM
