import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
import pandas as pd
import itertools as it
from collections import defaultdict
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.axis as axis
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import pylab as plot

#********************************
#Notes
#********************************

#Plots 2 item:
#1) a pairwise energy for indiviual peeling strands, and non-peeling strands
#2)
#********************************
#End Notes
#********************************
# only change the below reading_file
axis_Label_font_size = 24
legend_font_size = 18
axis_number_font_size = 20

PointSizes = 8
MarkeredgeWidth = 2
ConnectionLineSizes = 2
Calc_Linestyle = 'none'  #'-' =  solid, '--' = dashed, ':'= dotted
Exp_Linestyle = '-'

set_dpi=300

Ne_solute = "Ne"
Rn_solute = "Rn"

figure_name = f'free_energy_nobel_Ne_Rn.pdf'

Color_for_Plot_Data_calc = 'r'
Color_for_Plot_Linnemann_calc =  'k'
Color_for_Plot_NIST_exp =  'b'
Color_for_Plot_Lewis_exp = '#00FF00' #'olivedrab', #00FFFF, '#00FF00'
Color_for_Plot_Crovetto_exp = '#00FF00' #'olivedrab', #00FFFF, '#00FF00'

marker_gomc = 's'
marker_Linnemann = 'o'
marker_Lewis = 'D'
marker_Crovetto = '^'

#**************
# T vs density plot ranges (start)
#*************
Ne_y_column_iter = 0
Ne_Free_energy_cal_per_mol_min = 2
Ne_Free_energy_cal_per_mol_max = 4
Ne_solute = "Ne"

Rn_y_column_iter = 0
Rn_Free_energy_cal_per_mol_min = 0
Rn_Free_energy_cal_per_mol_max = 3
Rn_solute = "Rn"

T_min = 270
T_max = 390

Free_energy_cal_per_mol_major = 1
Free_energy_cal_per_mol_minor = 0.2

T_ticks_major = 20
T_ticks_minor = 5

Inverse_T_scalar_10_to_Exp = 4
#**************
# P vs T plot ranges (end)
#*************


#****************************************
#Plot Number 1  (temp vs density) (start)
#****************************************

# Plotting curve data below

fig1, ax1 = plt.subplots(2, sharex=True)

Free_energy_calc_file_reading_name = f"analysis_avg_std_of_replicates_box_0.txt"

Ne_Free_energy_calc_file_Linnemann_reading_name = f"Linnemann_et_al_Rn_Mick_et_al/" \
                                               f"avg_std_free_energy_data_waterTIP4P_Linnemann_et_al_Rn_Mick_et_al_" \
                                               f"{Ne_solute}.csv"
Ne_Free_energy_exp_file_NIST_reading_name = f"NIST_website/avg_std_free_energy_data_NIST_website_{Ne_solute}.csv"

Rn_Free_energy_calc_file_Linnemann_reading_name = f"Linnemann_et_al_Rn_Mick_et_al/" \
                                               f"avg_std_free_energy_data_waterTIP4P_Linnemann_et_al_Rn_Mick_et_al_" \
                                               f"{Rn_solute}.csv"
Rn_Free_energy_exp_file_NIST_reading_name = f"NIST_website/avg_std_free_energy_data_NIST_website_{Rn_solute}.csv"

Rn_Free_energy_exp_file_Lewis_reading_name = f"Lewis_et_al/" \
                                             f"avg_std_free_energy_data_exp_Lewis_et_al_" \
                                             f"{Rn_solute}.csv"

Ne_Free_energy_exp_file_Crovetto_reading_name = f"Crovetto_et_al/" \
                                                f"avg_std_free_energy_data_exp_Crovetto_et_al_" \
                                                f"{Ne_solute}.csv"

#********************************
#  File importing
#********************************

#data import Our data calcd
Free_energy_calc_data = pd.read_csv(Free_energy_calc_file_reading_name,  sep='\s+',
                                    header=0,
                                    )
Free_energy_calc_data_df = pd.DataFrame(Free_energy_calc_data)

Ne_Free_energy_calc_data_df = Free_energy_calc_data_df.loc[lambda df: df['solute'] == Ne_solute]
Rn_Free_energy_calc_data_df = Free_energy_calc_data_df.loc[lambda df: df['solute'] == Rn_solute]


Ne_Temp_calc_data = Ne_Free_energy_calc_data_df.loc[:, "temp_K"].values.tolist()
Rn_Temp_calc_data = Rn_Free_energy_calc_data_df.loc[:, "temp_K"].values.tolist()

Ne_avg_free_energy_calc_free = \
    Ne_Free_energy_calc_data_df.loc[:, "dFE_MBAR_kcal_per_mol"].values.tolist()

Ne_std_dev_free_energy_calc_free = \
    Ne_Free_energy_calc_data_df.loc[:, "dFE_MBAR_std_kcal_per_mol"].values.tolist()

Rn_avg_free_energy_calc_free = \
    Rn_Free_energy_calc_data_df.loc[:, "dFE_MBAR_kcal_per_mol"].values.tolist()

Rn_std_dev_free_energy_calc_free = \
    Rn_Free_energy_calc_data_df.loc[:, "dFE_MBAR_std_kcal_per_mol"].values.tolist()

# data import Linnemann data calcd
Ne_Free_energy_Linnemann_calc_data = pd.read_csv(Ne_Free_energy_calc_file_Linnemann_reading_name, sep=',')
Ne_Free_energy_Linnemann_calc_data_df = pd.DataFrame(Ne_Free_energy_Linnemann_calc_data)

Ne_Temp_Linnemann_calc_data = Ne_Free_energy_Linnemann_calc_data_df.loc[:, "temp_K"].values.tolist()
Ne_avg_free_energy_Linnemann_calc_free = Ne_Free_energy_Linnemann_calc_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
Ne_std_dev_free_energy_Linnemann_calc_free = Ne_Free_energy_Linnemann_calc_data_df.loc[:, "std_dev_MBAR_kcal_per_mol"].values.tolist()

Rn_Free_energy_Linnemann_calc_data = pd.read_csv(Rn_Free_energy_calc_file_Linnemann_reading_name, sep=',')
Rn_Free_energy_Linnemann_calc_data_df = pd.DataFrame(Rn_Free_energy_Linnemann_calc_data)

Rn_Temp_Linnemann_calc_data = Rn_Free_energy_Linnemann_calc_data_df.loc[:, "temp_K"].values.tolist()
Rn_avg_free_energy_Linnemann_calc_free = Rn_Free_energy_Linnemann_calc_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
Rn_std_dev_free_energy_Linnemann_calc_free = Rn_Free_energy_Linnemann_calc_data_df.loc[:, "std_dev_MBAR_kcal_per_mol"].values.tolist()


# data import NIST experimental
Ne_Free_energy_NIST_exp_data = pd.read_csv(Ne_Free_energy_exp_file_NIST_reading_name, sep=',')
Ne_Free_energy_NIST_exp_data_df = pd.DataFrame(Ne_Free_energy_NIST_exp_data)

Ne_Temp_NIST_exp_data = Ne_Free_energy_NIST_exp_data_df.loc[:, "temp_K"].values.tolist()
Ne_avg_free_energy_NIST_exp_free = Ne_Free_energy_NIST_exp_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
#Ne_std_dev_free_energy_NIST_exp_free = Ne_Free_energy_NIST_exp_data_df.loc[:, 'std_dev_MBAR_kcal_per_mol'].values.tolist()

Rn_Free_energy_NIST_exp_data = pd.read_csv(Rn_Free_energy_exp_file_NIST_reading_name, sep=',')
Rn_Free_energy_NIST_exp_data_df = pd.DataFrame(Rn_Free_energy_NIST_exp_data)

Rn_Temp_NIST_exp_data = Rn_Free_energy_NIST_exp_data_df.loc[:, "temp_K"].values.tolist()
Rn_avg_free_energy_NIST_exp_free = Rn_Free_energy_NIST_exp_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
#Rn_std_dev_free_energy_NIST_exp_free = Rn_Free_energy_NIST_exp_data_df.loc[:, 'std_dev_MBAR_kcal_per_mol'].values.tolist()


# data import Crovetto Ne experimental
Ne_Free_energy_Crovetto_exp_data = pd.read_csv(Ne_Free_energy_exp_file_Crovetto_reading_name, sep=',')
Ne_Free_energy_Crovetto_exp_data_df = pd.DataFrame(Ne_Free_energy_Crovetto_exp_data)

Ne_Temp_Crovetto_exp_data = Ne_Free_energy_Crovetto_exp_data_df.loc[:, "temp_K"].values.tolist()
Ne_avg_free_energy_Crovetto_exp_free = Ne_Free_energy_Crovetto_exp_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
#Ne_std_dev_free_energy_Crovetto_exp_free = Ne_Free_energy_Crovetto_exp_data_df.loc[:, 'std_dev_MBAR_kcal_per_mol'].values.tolist()

# data import Lewis Re experimental
Rn_Free_energy_Lewis_exp_data = pd.read_csv(Rn_Free_energy_exp_file_Lewis_reading_name, sep=',')
Rn_Free_energy_Lewis_exp_data_df = pd.DataFrame(Rn_Free_energy_Lewis_exp_data)

Rn_Temp_Lewis_exp_data = Rn_Free_energy_Lewis_exp_data_df.loc[:, "temp_K"].values.tolist()
Rn_avg_free_energy_Lewis_exp_free = Rn_Free_energy_Lewis_exp_data_df.loc[:, "avg_MBAR_kcal_per_mol"].values.tolist()
Rn_std_dev_free_energy_Lewis_exp_free = Rn_Free_energy_Lewis_exp_data_df.loc[:, 'std_dev_MBAR_kcal_per_mol'].values.tolist()

# ********************************
#  End File importing
# ********************************




#****************************************
#Plot Number 1  (temp vs density) (start)
#****************************************

# Plotting curve data below

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(1, 1, 1)


#for tick in ax1.xaxis.get_ticklabels():
#    tick.set_fontname('Arial')
#for tick in ax1.yaxis.get_ticklabels():
#    tick.set_fontname('Arial')

ax1[1].set_xlabel('Temperature (K)', fontname="Arial", fontsize=axis_Label_font_size)
ax1[1].set_ylabel('                $\u0394_{}$G (kcal/mol)', fontname="Arial", fontsize=axis_Label_font_size)

#avg_free_energy_calc_free_label = 'This Work'
#avg_free_energy_Linnemann_calc_free_label = "Calculated: Linnemann et. al. \nSimulations"
#avg_free_energy_NIST_exp_free_label = "Calculated: NIST Website \nExperimental Data"
#avg_free_energy_Lewis_exp_free_label = "Calculated: Lewis et al. \nExperimental Data"
#avg_free_energy_Crovetto_exp_free_label = "Experiment: Crovetto et al."


ax1[0].plot(Ne_Temp_NIST_exp_data,
         Ne_avg_free_energy_NIST_exp_free,
         color=Color_for_Plot_NIST_exp,
         marker=None,
         linestyle=Exp_Linestyle,
         markersize=PointSizes,
         markeredgewidth=MarkeredgeWidth,
         linewidth=ConnectionLineSizes,
         fillstyle='none',
         #label=avg_free_energy_NIST_exp_free_label,
         )
ax1[0].plot(Ne_Temp_Crovetto_exp_data,
         Ne_avg_free_energy_Crovetto_exp_free,
         color=Color_for_Plot_Crovetto_exp,
         marker=marker_Crovetto,
         linestyle=Calc_Linestyle,
         markersize=PointSizes,
         markeredgewidth=MarkeredgeWidth,
         linewidth=ConnectionLineSizes,
         fillstyle='full',
         #capsize=10,
         #label=avg_free_energy_Crovetto_exp_free_label,
         )
ax1[0].errorbar(Ne_Temp_Linnemann_calc_data,
             Ne_avg_free_energy_Linnemann_calc_free ,
             yerr=Ne_std_dev_free_energy_Linnemann_calc_free,
             color=Color_for_Plot_Linnemann_calc,
             marker=marker_Linnemann,
             linestyle=Calc_Linestyle,
             markersize=PointSizes,
             markeredgewidth=MarkeredgeWidth,
             linewidth=ConnectionLineSizes,
             fillstyle='none',
             capsize=10,
             #label=avg_free_energy_Linnemann_calc_free_label
             )
ax1[0].errorbar(Ne_Temp_calc_data,
             Ne_avg_free_energy_calc_free,
             yerr=Ne_std_dev_free_energy_calc_free,
             color=Color_for_Plot_Data_calc,
             marker=marker_gomc,
             linestyle=Calc_Linestyle,
             markersize=PointSizes,
             markeredgewidth=MarkeredgeWidth,
             linewidth=ConnectionLineSizes,
             fillstyle='none',
             capsize=10,
             #label=avg_free_energy_calc_free_label,
             )



ax1[1].plot(Rn_Temp_NIST_exp_data,
         Rn_avg_free_energy_NIST_exp_free,
         color=Color_for_Plot_NIST_exp,
         marker=None,
         linestyle=Exp_Linestyle,
         markersize=PointSizes,
         markeredgewidth=MarkeredgeWidth,
         linewidth=ConnectionLineSizes,
         fillstyle='none',
         #label=avg_free_energy_NIST_exp_free_label,
         )
ax1[1].plot(Rn_Temp_Lewis_exp_data ,
         Rn_avg_free_energy_Lewis_exp_free,
         color=Color_for_Plot_Lewis_exp,
         marker=marker_Lewis,
         linestyle=Calc_Linestyle,
         markersize=PointSizes,
         markeredgewidth=MarkeredgeWidth,
         linewidth=ConnectionLineSizes,
         fillstyle='full',
         #capsize=10,
         #label=avg_free_energy_Lewis_exp_free_label,
         )
'''
ax1[1].errorbar(Rn_Temp_Lewis_exp_data ,
                Rn_avg_free_energy_Lewis_exp_free,
                yerr=Rn_std_dev_free_energy_Lewis_exp_free,
                color=Color_for_Plot_Lewis_exp,
                marker=marker_Lewis,
                linestyle=Calc_Linestyle,
                markersize=PointSizes-1,
                markeredgewidth=MarkeredgeWidth,
                linewidth=ConnectionLineSizes,
                fillstyle='full',
                capsize=10,
                # label=avg_free_energy_Lewis_exp_free_label,
             )
'''
ax1[1].errorbar(Rn_Temp_Linnemann_calc_data,
             Rn_avg_free_energy_Linnemann_calc_free ,
             yerr=Rn_std_dev_free_energy_Linnemann_calc_free,
             color=Color_for_Plot_Linnemann_calc,
             marker=marker_Linnemann,
             linestyle=Calc_Linestyle,
             markersize=PointSizes,
             markeredgewidth=MarkeredgeWidth,
             linewidth=ConnectionLineSizes,
             fillstyle='none',
             capsize=10,
             #label=avg_free_energy_Linnemann_calc_free_label
             )
ax1[1].errorbar(Rn_Temp_calc_data,
             Rn_avg_free_energy_calc_free,
             yerr=Rn_std_dev_free_energy_calc_free,
             color=Color_for_Plot_Data_calc,
             marker=marker_gomc,
             linestyle=Calc_Linestyle,
             markersize=PointSizes,
             markeredgewidth=MarkeredgeWidth,
             linewidth=ConnectionLineSizes,
             fillstyle='none',
             capsize=10,
             #label=avg_free_energy_calc_free_label,
             )


ax1[0].text(275, 3.58, "Ne", weight='bold', size=legend_font_size)
ax1[1].text(275, 2.36, "Rn", weight='bold', size=legend_font_size)

major_yticks = np.arange(0, 20+0.001, Free_energy_cal_per_mol_major)
major_xticks = np.arange(T_min, T_max+0.001, T_ticks_major)

minor_yticks = np.arange(0, 20+0.001, Free_energy_cal_per_mol_minor )
minor_xticks = np.arange(T_min, T_max+0.001, T_ticks_minor)


#plt.gca().set_xlim(left=2, right=105)

ax1[0].set_xticks(major_xticks)
ax1[0].set_xticks(minor_xticks, minor=True)
ax1[0].set_yticks(major_yticks)
ax1[0].set_yticks(minor_yticks, minor=True)

pad_whitepace_for_numbering = 10
ax1[0].tick_params(axis='both', which='major', length=8, width=2, direction='in',
                   labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)
ax1[0].tick_params(axis='both', which='minor', length=4, width=1, direction='in',
                   labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)

ax1[1].set_xticks(major_xticks)
ax1[1].set_xticks(minor_xticks, minor=True)
ax1[1].set_yticks(major_yticks)
ax1[1].set_yticks(minor_yticks, minor=True)

ax1[1].tick_params(axis='both', which='major', length=8, width=2, direction='in',
                   labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)
ax1[1].tick_params(axis='both', which='minor', length=4, width=1, direction='in',
                   labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)

#legend1 = ax1.legend(loc='upper center', shadow=True, fontsize=legend_font_size )

#frame1 = legend1.get_frame()
#frame1.set_facecolor('0.90')
#ax1[0, y_column_iter].tight_layout()  # centers layout nice for final paper
# plt.gcf().subplots_adjust(bottom=0.15) # moves plot up so x label not cutoff

ax1[0].set_xlim(T_min, T_max+0.001)  # set plot range on x axis
ax1[1].set_xlim(T_min, T_max+0.001)  # set plot range on x axis


ax1[0].set_ylim(Ne_Free_energy_cal_per_mol_min, Ne_Free_energy_cal_per_mol_max)  # set plot range on x axis
#ax1[0].legend(ncol=1, loc='upper left', fontsize=legend_font_size,
#           prop={'family': 'Arial', 'size': legend_font_size})
ax1[1].set_ylim(Rn_Free_energy_cal_per_mol_min, Rn_Free_energy_cal_per_mol_max)  # set plot range on x axis
#ax1[1].legend(ncol=1, loc='upper left', fontsize=legend_font_size,
#           prop={'family': 'Arial', 'size': legend_font_size})
#plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
plt.gcf().subplots_adjust(left=0.12, bottom=0.18, right=0.94, top=0.95, wspace=None, hspace=None) # moves plot  so x label not cutoff
plt.show()
fig1.savefig(figure_name)

plt.close()

#****************************************
#Plot Number 1  (temp vs density) (end)
#****************************************




