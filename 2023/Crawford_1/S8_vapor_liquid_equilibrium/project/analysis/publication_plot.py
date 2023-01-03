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
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as plot

#********************************
#********************************
#Notes (start)
#********************************
#********************************

#Plots 2 item:
#1) T vs Rho
#2) ln(P) vs 1/T
#********************************
#********************************
#End Notes (end)
#********************************
#********************************

#********************************
#********************************
# user variables (start)
#********************************
#********************************

axis_Label_font_size = 24
legend_font_size = 18
axis_number_font_size = 20

PointSizes = 8
Critical_PointSizes = 12
Boiling_point_PointSizes = 8
MarkerWidth = 2
ConnectionLineSizes = 2
Reg_Rho_kg_per_m_cubed_Linestyle = '--'  #'-' =  solid, '--' = dashed, ':'= dotted
Critical_Rho_kg_per_m_cubed_Linestyle = None

set_dpi=300

#**************
# T vs Rho plot ranges (start)
#*************
Rho_kg_per_m_cubed_kg_per_m_cubed_min = 0
Rho_kg_per_m_cubed_kg_per_m_cubed_max = 700

T_from_Rho_kg_per_m_cubed_min = 400
T_from_Rho_kg_per_m_cubed_max = 700

Rho_kg_per_m_cubed_ticks_major = 100
Rho_kg_per_m_cubed_ticks_minor = 20

T_from_Rho_kg_per_m_cubed_ticks_major = 100
T_from_Rho_kg_per_m_cubed_ticks_minor = 20
#**************
# T vs Rho plot ranges (end)
#*************


#**************
# ln(P) vs 1/T plot ranges (start)
#*************
ln_P_min = -2
ln_P_max = 4

ln_P_ticks_major = 1
ln_P_ticks_minor = 0.2

inverse_T_vs_ln_P_min = 14
inverse_T_vs_ln_P_max = 28

inverse_T_vs_ln_P_ticks_major = 2
inverse_T_vs_ln_P_ticks_minor = 0.5

Inverse_T_scalar_10_to_Exp = 4
#**************
# ln(P) vs 1/T plot ranges (end)
#*************

Rho_kg_per_m_cubed_file_reading_name_liq = "analysis_avg_std_of_replicates_box_liq.txt"
Rho_kg_per_m_cubed_file_reading_name_vap = "analysis_avg_std_of_replicates_box_vap.txt"
Critical_calcd_data_file_reading_name = "analysis_critical_points_avg_std_of_replicates.txt"
Boiling_calcd_data_file_reading_name = "analysis_boiling_point_avg_std_of_replicates.txt"

Density_and_Claperon_file_saving_name = "S8_Density_and_Claperon.pdf"

Color_for_Plot_Data = 'r' # red
Color_for_Plot_Critical_Data_calc =  'k'
Color_for_Plot_boiling_point_calc = 'k'

marker = "s"
critical_marker = "*"
boiling_point_marker = "o"
error_bar_fmt = ''
error_bar_ecolor = 'k'
error_bar_elinewidth = 2
error_bar_capsize = 4
fill_simulation_point = 'none'
fill_critical_point = 'full'
fill_boiling_point = 'full'

#********************************
#********************************
# user variables (start)
#********************************
#********************************

#********************************
#  File importing  (start)
#********************************

#data import
data_file_liq_df = pd.DataFrame(pd.read_csv(Rho_kg_per_m_cubed_file_reading_name_liq,  sep='\s+'))

Temp_K_mean_from_Rho_kg_per_m_cubed_liq = data_file_liq_df.loc[:, 'temp_K'].values.tolist()

Rho_kg_per_m_cubed_mean_liq = data_file_liq_df.loc[:, 'Rho_kg_per_m_cubed'].values.tolist()
Rho_kg_per_m_cubed_std_liq = data_file_liq_df.loc[:, 'Rho_std_kg_per_m_cubed'].values.tolist()

P_bar_mean_liq = data_file_liq_df.loc[:, 'P_bar'].values.tolist()
P_bar_std_liq = data_file_liq_df.loc[:, 'P_std_bar'].values.tolist()
P_bar_mean_liq_max = [P_bar_mean_liq[i] + P_bar_std_liq[i] for i in range(0, len(P_bar_mean_liq))]
P_bar_mean_liq_min = [P_bar_mean_liq[i] - P_bar_std_liq[i] for i in range(0, len(P_bar_mean_liq))]


data_file_vap_df  = pd.DataFrame(pd.read_csv(Rho_kg_per_m_cubed_file_reading_name_vap,  sep='\s+'))

Temp_K_mean_from_Rho_kg_per_m_cubed_vap =data_file_vap_df.loc[:, 'temp_K'].values.tolist()

Rho_kg_per_m_cubed_mean_vap = data_file_vap_df.loc[:, 'Rho_kg_per_m_cubed'].values.tolist()
Rho_kg_per_m_cubed_std_vap = data_file_vap_df.loc[:, 'Rho_std_kg_per_m_cubed'].values.tolist()

P_bar_mean_vap = data_file_vap_df.loc[:, 'P_bar'].values.tolist()
P_bar_std_vap = data_file_vap_df.loc[:, 'P_std_bar'].values.tolist()
P_bar_mean_vap_max = [P_bar_mean_vap[i] + P_bar_std_vap[i] for i in range(0, len(P_bar_mean_vap))]
P_bar_mean_vap_min = [P_bar_mean_vap[i] - P_bar_std_vap[i] for i in range(0, len(P_bar_mean_vap))]

# get the Claperon data
Claperon_inverse_Temp_K_calc = [
    1/i*(10**(Inverse_T_scalar_10_to_Exp ))  for i in Temp_K_mean_from_Rho_kg_per_m_cubed_vap
]
Claperon_ln_P_calc = [np.log(i) for i in P_bar_mean_vap]

#get std dev
Claperon_ln_P_std_calc = []
for j in range(0, len(P_bar_mean_vap)):
    log_value_max = np.maximum(abs(np.log(P_bar_mean_vap[j]) - np.log(P_bar_mean_vap_min[j])),
                               abs(np.log(P_bar_mean_vap_max[j]) - np.log(P_bar_mean_vap[j]))
                               )

    Claperon_ln_P_std_calc.append(log_value_max)


# get the critical data
Critical_calcd_data = pd.read_csv(Critical_calcd_data_file_reading_name,  sep='\s+')
Critical_calcd_data_df = pd.DataFrame(Critical_calcd_data)

Critical_Temp_K_calc_mean = Critical_calcd_data_df.loc[0, 'Tc_K']
Critical_Temp_K_calc_std = Critical_calcd_data_df.loc[0, 'Tc_std_K']
Critical_inverse_Temp_K_calc_mean = 1 / Critical_Temp_K_calc_mean * (10**(Inverse_T_scalar_10_to_Exp))
Critical_inverse_Temp_K_calc_max = \
    1 / (Critical_Temp_K_calc_mean - Critical_Temp_K_calc_std) * (10**(Inverse_T_scalar_10_to_Exp))
Critical_inverse_Temp_K_calc_min = \
    1 / (Critical_Temp_K_calc_mean + Critical_Temp_K_calc_std) * (10**(Inverse_T_scalar_10_to_Exp))
Critical_inverse_Temp_K_calc_std = \
    Critical_inverse_Temp_K_calc_mean * ((Critical_Temp_K_calc_std/Critical_Temp_K_calc_mean)**2)**0.5

Critical_Rho_kg_per_m_cubed_calc_mean = Critical_calcd_data_df.loc[0, 'Rho_c_kg_per_m_cubed']
Critical_Rho_kg_per_m_cubed_calc_std = Critical_calcd_data_df.loc[0, 'Rho_c_std_kg_per_m_cubed']

Critical_P_bar_calc_mean = Critical_calcd_data_df.loc[0, 'Pc_bar']
Critical_P_bar_calc_std = Critical_calcd_data_df.loc[0, 'Pc_std_bar']
Critical_P_bar_calc_max = Critical_P_bar_calc_mean + Critical_P_bar_calc_std
Critical_P_bar_calc_min = Critical_P_bar_calc_mean - Critical_P_bar_calc_std

Critical_ln_P_bar_calc_mean = np.log(Critical_P_bar_calc_mean)
Critical_ln_P_bar_calc_std = np.maximum(abs(np.log(Critical_P_bar_calc_mean) - np.log(Critical_P_bar_calc_min)),
                               abs(np.log(Critical_P_bar_calc_max) - np.log(Critical_P_bar_calc_mean))
                               )


# get the boiling point data
Boiling_calcd_data = pd.read_csv(Boiling_calcd_data_file_reading_name,  sep='\s+')
Boiling_calcd_data_df = pd.DataFrame(Boiling_calcd_data)

Boiling_Temp_K_calc_mean = Boiling_calcd_data_df.loc[0, 'Tbp_K']
Boiling_Temp_K_calc_std = Boiling_calcd_data_df.loc[0, 'Tbp_std_K']

Boiling_pressure_bar = Boiling_calcd_data_df.loc[0, 'Pbp_bar']


Boiling_inverse_Temp_K_calc_mean = 1 / Boiling_Temp_K_calc_mean*(10**(Inverse_T_scalar_10_to_Exp ))
Boiling_inverse_Temp_K_calc_std = 1 / Boiling_Temp_K_calc_std*(10**(Inverse_T_scalar_10_to_Exp ))
Boiling_inverse_Temp_K_calc_max = \
    1 / (Boiling_Temp_K_calc_mean - Boiling_Temp_K_calc_std) * (10**(Inverse_T_scalar_10_to_Exp))
Boiling_inverse_Temp_K_calc_min = \
    1 / (Boiling_Temp_K_calc_mean + Boiling_Temp_K_calc_std) * (10**(Inverse_T_scalar_10_to_Exp))
Boiling_inverse_Temp_K_calc_std = \
    Boiling_inverse_Temp_K_calc_mean * ((Boiling_Temp_K_calc_std/Boiling_Temp_K_calc_mean)**2)**0.5
Boiling_ln_pressure_bar = np.log(Boiling_pressure_bar)

Hv_calc_at_Claus_Clap_mean = Boiling_calcd_data_df.loc[0, 'Hv_kJ_per_mol_Claus_Clap']
Hv_calc_at_Claus_Clap_std = Boiling_calcd_data_df.loc[0, 'Hv_std_kJ_per_mol_Claus_Clap']

print(f"Boiling_Temp_K_calc_mean = {Boiling_Temp_K_calc_mean}")
print(f"Boiling_Temp_K_calc_std = {Boiling_Temp_K_calc_std}")
print(f"Boiling_pressure_bar = {Boiling_pressure_bar}")
print(f"Boiling_inverse_Temp_K_calc_mean = {Boiling_inverse_Temp_K_calc_mean}")
print(f"Boiling_inverse_Temp_K_calc_std = {Boiling_inverse_Temp_K_calc_std}")
print(f"Boiling_ln_pressure_bar = {Boiling_ln_pressure_bar}")

# ********************************
#  File importing (end)
# ********************************

#****************************************
#Plot Number 1  (temp vs density) (start)
#****************************************

# Plotting curve data below
fig = plt.figure()
axDensity = plt.subplot(111)

divider = make_axes_locatable(axDensity)
divider_pad = 1.4
axClausius = divider.append_axes("right",
                                 size=4.56,
                                 pad=divider_pad
                                 )
fig.set_figwidth(12)
fig.set_figheight(5)

#fig.subplots_adjust(wspace=0, hspace=0)

axDensity.set_xlabel(r'$\rho$ (kg/m$^3$)', fontname="Arial", fontsize=axis_Label_font_size)
axDensity.set_ylabel('Temperature (K)', fontname="Arial", fontsize=axis_Label_font_size)

File_label_Rho_kg_per_m_cubed = 'Simulated'
File_label_Rho_kg_per_m_cubed_Critical_calc = 'Simulated\nCritical Point'
File_label_Rho_kg_per_m_cubed_Critical_exp = 'Exp'


axDensity.errorbar(Rho_kg_per_m_cubed_mean_liq,
             Temp_K_mean_from_Rho_kg_per_m_cubed_liq,
             #xerr=Rho_kg_per_m_cubed_std_liq,
             #yerr=None,
             color=Color_for_Plot_Data,
             marker=marker,
             linestyle='none',
             markersize=PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Rho_kg_per_m_cubed,
             fillstyle=fill_simulation_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axDensity.errorbar(Rho_kg_per_m_cubed_mean_vap,
             Temp_K_mean_from_Rho_kg_per_m_cubed_vap,
             #xerr=Rho_kg_per_m_cubed_std_vap,
             #yerr=None,
             color=Color_for_Plot_Data,
             marker=marker,
             linestyle='none',
             markersize=PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=None,
             fillstyle=fill_simulation_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axDensity.errorbar(Critical_Rho_kg_per_m_cubed_calc_mean,
             Critical_Temp_K_calc_mean,
             #xerr=Critical_Rho_kg_per_m_cubed_calc_std,
             #yerr=Critical_Temp_K_calc_std,
             color=Color_for_Plot_Critical_Data_calc,
             marker=critical_marker,
             linestyle='none',
             markersize=Critical_PointSizes ,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Rho_kg_per_m_cubed_Critical_calc,
             fillstyle=fill_critical_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axDensity.text(630, 660, "A", weight='bold', size=legend_font_size)

major_xticks = np.arange(Rho_kg_per_m_cubed_kg_per_m_cubed_min,
                         Rho_kg_per_m_cubed_kg_per_m_cubed_max+0.001,
                         Rho_kg_per_m_cubed_ticks_major)
major_yticks = np.arange(T_from_Rho_kg_per_m_cubed_min,
                         T_from_Rho_kg_per_m_cubed_max+0.001,
                         T_from_Rho_kg_per_m_cubed_ticks_major)

minor_xticks = np.arange(Rho_kg_per_m_cubed_kg_per_m_cubed_min,
                         Rho_kg_per_m_cubed_kg_per_m_cubed_max+0.001,
                         Rho_kg_per_m_cubed_ticks_minor )
minor_yticks = np.arange(T_from_Rho_kg_per_m_cubed_min,
                         T_from_Rho_kg_per_m_cubed_max+0.001,
                         T_from_Rho_kg_per_m_cubed_ticks_minor)

#plt.gca().set_xlim(left=2, right=105)

axDensity.set_xticks(major_xticks)
axDensity.set_xticks(minor_xticks, minor=True)
axDensity.set_yticks(major_yticks)
axDensity.set_yticks(minor_yticks, minor=True)

pad_whitepace_for_numbering = 10
axDensity.tick_params(axis='both', which='major', length=8, width=2, direction='in',
                labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)
axDensity.tick_params(axis='both', which='minor', length=4, width=1, direction='in',
                labelsize=axis_number_font_size, top=True, right=True,
                   pad=pad_whitepace_for_numbering)

#legend1 = ax1.legend(loc='lower center', shadow=False, fontsize=legend_font_size )

#frame1 = legend1.get_frame()
#frame1.set_facecolor('0.90')

# plt.gcf().subplots_adjust(bottom=0.15) # moves plot up so x label not cutoff
axDensity.set_xlim(Rho_kg_per_m_cubed_kg_per_m_cubed_min, Rho_kg_per_m_cubed_kg_per_m_cubed_max+0.0001)  # set plot range on x axis
axDensity.set_ylim(T_from_Rho_kg_per_m_cubed_min, T_from_Rho_kg_per_m_cubed_max+0.0001)  # set plot range on x axis
#plt.gcf().subplots_adjust(left=0.15, bottom=None, right=0.95, top=None, wspace=None, hspace=None) # moves plot  so x label not cutoff

#plt.legend(ncol=1,loc='lower center', fontsize=legend_font_size, prop={'family':'Arial','size': legend_font_size})


#****************************************
#Plot Number 1  (temp vs density) (end)
#****************************************





#****************************************
# Plot Number 2  ln(P) vs 1/T (start)
#****************************************

# Plotting curve data below


x = str(-Inverse_T_scalar_10_to_Exp)
xlabel_inter = f'1/T * 10$^[{-Inverse_T_scalar_10_to_Exp}]$ (1/K)'
mod_x_label = ""
for char_iter in xlabel_inter:
    if char_iter == '[':
        mod_x_label += '{'
    elif char_iter == ']':
        mod_x_label += '}'
    else:
        mod_x_label += char_iter

axClausius.set_xlabel(mod_x_label, fontname="Arial", fontsize=axis_Label_font_size)
axClausius.set_ylabel('ln(P/bar)', fontname="Arial", fontsize=axis_Label_font_size)

File_label_Rho_kg_per_m_cubed = 'Simulated'
File_label_Rho_kg_per_m_cubed_Critical_calc = 'Simulated\nCritical Point'
File_label_Rho_kg_per_m_cubed_Critical_exp = 'Exp'

axClausius.errorbar(Claperon_inverse_Temp_K_calc,
             Claperon_ln_P_calc,
             #xerr=None,
             #yerr=Claperon_ln_P_std_calc,
             color=Color_for_Plot_Data,
             marker=marker,
             linestyle='none',
             markersize=PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Rho_kg_per_m_cubed,
             fillstyle=fill_simulation_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axClausius.errorbar(Critical_inverse_Temp_K_calc_mean,
             Critical_ln_P_bar_calc_mean,
             #xerr=Critical_inverse_Temp_K_calc_std,
             #yerr=Critical_ln_P_bar_calc_std,
             color=Color_for_Plot_Critical_Data_calc,
             marker=critical_marker,
             linestyle='none',
             markersize=Critical_PointSizes ,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Rho_kg_per_m_cubed_Critical_calc,
             fillstyle=fill_critical_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axClausius.errorbar(Boiling_inverse_Temp_K_calc_mean,
             Boiling_ln_pressure_bar,
             #xerr=Boiling_inverse_Temp_K_calc_std,
             color=Color_for_Plot_boiling_point_calc,
             marker=boiling_point_marker,
             linestyle='none',
             markersize=Boiling_point_PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Rho_kg_per_m_cubed_Critical_calc,
             fillstyle=fill_boiling_point,
             fmt=error_bar_fmt,
             ecolor=error_bar_ecolor,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

axClausius.text(26.5, 3.2, "B", weight='bold', size=legend_font_size)

'''
bp_decimal_rounding = 1
T_bp_mean_rounded = np.round(Boiling_Temp_K_calc_mean, decimals=bp_decimal_rounding)
T_bp_std_rounded = np.round(Boiling_Temp_K_calc_std, decimals=bp_decimal_rounding)
plot_printed_T_bp_start_of_string = "T$_{BP}$ = "
plot_printed_T_bp_end_of_string = f"{T_bp_mean_rounded} +/- {T_bp_std_rounded}"
plot_printed_T_bp_combined_string = plot_printed_T_bp_start_of_string + plot_printed_T_bp_end_of_string
axClausius.text(14.5, -1.65, plot_printed_T_bp_combined_string,
                color=Color_for_Plot_boiling_point_calc,
                weight='bold', size=legend_font_size)
'''
major_xticks = np.arange(inverse_T_vs_ln_P_min, inverse_T_vs_ln_P_max+0.001, inverse_T_vs_ln_P_ticks_major)
major_yticks = np.arange(ln_P_min , ln_P_max+0.0001, ln_P_ticks_major)

minor_xticks = np.arange(inverse_T_vs_ln_P_min, inverse_T_vs_ln_P_max+0.001, inverse_T_vs_ln_P_ticks_minor)
minor_yticks = np.arange(ln_P_min , ln_P_max+0.0001, ln_P_ticks_minor)


#plt.gca().set_xlim(left=2, right=105)

axClausius.set_xticks(major_xticks)
axClausius.set_xticks(minor_xticks, minor=True)
axClausius.set_yticks(major_yticks)
axClausius.set_yticks(minor_yticks, minor=True)

pad_whitepace_for_numbering = 10
axClausius.tick_params(axis='both', which='major', length=8, width=2, direction='in',
                       labelsize=axis_number_font_size, top=True, right=True,
                       pad=pad_whitepace_for_numbering)
axClausius.tick_params(axis='both', which='minor', length=4, width=1, direction='in',
                       labelsize=axis_number_font_size, top=True, right=True,
                       pad=pad_whitepace_for_numbering)

#leg2 = ax2.legend(loc='upper right', shadow=True, fontsize=legend_font_size ,prop={'family':'Arial','size': legend_font_size})

# plt.gcf().subplots_adjust(bottom=0.15) # moves plot up so x label not cutoff
axClausius.set_xlim(inverse_T_vs_ln_P_min, inverse_T_vs_ln_P_max+0.001)  # set plot range on x axis
axClausius.set_ylim(ln_P_min, ln_P_max+0.0001)  # set plot range on y axis

plt.subplots_adjust(left=0.15, bottom=None, right=0.90, top=None, wspace=None, hspace=0) # moves plot  so x label not cutoff

#frame1 = legend1.get_frame()
#frame1.set_facecolor('0.90')

#plt.legend(ncol=1, loc='upper right', fontsize=legend_font_size, prop={'family':'Arial','size': legend_font_size})

plt.tight_layout()  # centers layout nice for final paper


plt.show()
fig.savefig(Density_and_Claperon_file_saving_name)
#****************************************
# Plot Number 2  ln(P) vs 1/T (end)
#****************************************
