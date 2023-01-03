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
import pylab as plot

#********************************
#********************************
#Notes (start)
#********************************
#********************************

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
MarkerWidth = 2
ConnectionLineSizes = 2
Reg_Rho_kg_per_m_cubed_Linestyle = '--'  #'-' =  solid, '--' = dashed, ':'= dotted
Critical_Rho_kg_per_m_cubed_Linestyle = None



#**************
# T vs Rho plot ranges (start)
#*************
pressure_bar_min = 0.01
pressure_bar_max = 10

adsorbed_molecules_per_unit_cell_min = 0.1
adsorbed_molecules_per_unit_cell_max = 10 # 10 orig

pressure_bar_ticks_major = 0.2
pressure_bar_ticks_minor = 0.1

adsorbed_molecules_per_unit_cell_major = 2
adsorbed_molecules_per_unit_cell_minor = 0.25
#**************
# T vs Rho plot ranges (end)
#*************


#**************
# log10(P) vs 1/T plot ranges (start)
#*************
log10_P_min = -2
log10_P_max = 4

log10_P_ticks_major = 1
log10_P_ticks_minor = 0.2

inverse_T_vs_log10_P_min = 14
inverse_T_vs_log10_P_max = 28

inverse_T_vs_log10_P_ticks_major = 2
inverse_T_vs_log10_P_ticks_minor = 0.5

Inverse_T_scalar_10_to_Exp = 4
#**************
# log10(P) vs 1/T plot ranges (end)
#*************

adsorption_reading_name = "analysis_avg_std_of_replicates_box_0.txt"
Yazaydin_Snurr_adsorption_sim_reading_name = "Yazaydin_Snurr_data/IRMOF_Yazaydin_Snurr_simulation_molecules_per_uc.csv"
Yazaydin_Snurr_adsorption_exp_reading_name = "Yazaydin_Snurr_data/IRMOF_Yazaydin_Snurr_experimental_data_molecules_per_uc.csv"

adsorbed_moleules_per_unit_cell_file_saving_name_CO2 = "adsorbed_CO2_per_unit_cell.pdf"

Color_for_Plot_Data = 'r' # red
Color_for_Plot_Yazaydin_Snurr_sim =  'k'  # black
Color_for_Plot_Yazaydin_Snurr_exp =  'g'  # black

marker = "s"
Yazaydin_Snurr_marker_sim = "o"
Yazaydin_Snurr_marker_exp = "D"
error_bar_fmt = ''
error_bar_ecolor = 'k'
error_bar_elinewidth = 2
error_bar_capsize = 4

#********************************
#********************************
# user variables (start)
#********************************
#********************************

#********************************
#  File importing  (start)
#********************************

# import GOMC data
data_file_zeolite = pd.DataFrame(pd.read_csv(adsorption_reading_name,  sep='\s+'))
data_file_zeolite_CO2 = data_file_zeolite.loc[lambda df: df['molecule_name'] == 'CO2']

P_bar_mean_CO2 = data_file_zeolite_CO2.loc[:, 'P_bar'].values.tolist()

log10_P_bar_calc_mean_CO2 = np.log10(P_bar_mean_CO2)

adsorpbed_mean_CO2 = data_file_zeolite_CO2.loc[:, 'ads_molecules_per_UC_mean'].values.tolist()
adsorpbed_std_CO2 = data_file_zeolite_CO2.loc[:, 'ads_molecules_per_UC_std'].values.tolist()


# import Yazaydin_Snurr et. al Simulation data
data_file_zeolite_Yazaydin_Snurr_sim = pd.DataFrame(pd.read_csv(Yazaydin_Snurr_adsorption_sim_reading_name,  sep=','))
data_file_zeolite_Yazaydin_Snurr_CO2_sim = data_file_zeolite_Yazaydin_Snurr_sim.loc[lambda df: df['molecule_name'] == 'CO2']

P_bar_mean_Yazaydin_Snurr_CO2_sim = data_file_zeolite_Yazaydin_Snurr_CO2_sim.loc[:, 'P_bar_mean'].values.tolist()

log10_P_bar_calc_mean_Yazaydin_Snurr_CO2_sim = np.log10(P_bar_mean_Yazaydin_Snurr_CO2_sim)

adsorpbed_mean_Yazaydin_Snurr_CO2_sim = data_file_zeolite_Yazaydin_Snurr_CO2_sim.loc[:, 'ads_molecules_per_UC_mean'].values.tolist()
adsorpbed_std_Yazaydin_Snurr_CO2_sim = data_file_zeolite_Yazaydin_Snurr_CO2_sim.loc[:, 'ads_molecules_per_UC_std'].values.tolist()


# import Yazaydin_Snurr et. al experimental data
data_file_zeolite_Yazaydin_Snurr_exp = pd.DataFrame(pd.read_csv(Yazaydin_Snurr_adsorption_exp_reading_name,  sep=','))
data_file_zeolite_Yazaydin_Snurr_CO2_exp = data_file_zeolite_Yazaydin_Snurr_exp.loc[lambda df: df['molecule_name'] == 'CO2']

P_bar_mean_Yazaydin_Snurr_CO2_exp = data_file_zeolite_Yazaydin_Snurr_CO2_exp.loc[:, 'P_bar_mean'].values.tolist()

log10_P_bar_calc_mean_Yazaydin_Snurr_CO2_exp = np.log10(P_bar_mean_Yazaydin_Snurr_CO2_exp)

adsorpbed_mean_Yazaydin_Snurr_CO2_exp = data_file_zeolite_Yazaydin_Snurr_CO2_exp.loc[:, 'ads_molecules_per_UC_mean'].values.tolist()
adsorpbed_std_Yazaydin_Snurr_CO2_exp = data_file_zeolite_Yazaydin_Snurr_CO2_exp.loc[:, 'ads_molecules_per_UC_std'].values.tolist()

# ********************************
#  File importing (end)
# ********************************

#****************************************
#Plot Number 1  (ln10 CO2 molecule / unit cell vs log10 pressure) (start)
#****************************************

# Plotting curve data below

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)


for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontname('Arial')
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontname('Arial')

plt.subplot(111, xscale="log", yscale="log")


plt.xlabel(r'Pressure (bar)', fontname="Arial", fontsize=axis_Label_font_size)
plt.ylabel('Molecules / unit cell', fontname="Arial", fontsize=axis_Label_font_size)

File_label = 'Simulated'
File_label_Yazaydin_Snurr_sim = 'Yazaydin_Snurr et. al \n simulation'
File_label_Yazaydin_Snurr_exp = 'Yazaydin_Snurr et. al \n experimental'


plt.errorbar(P_bar_mean_Yazaydin_Snurr_CO2_exp,
             adsorpbed_mean_Yazaydin_Snurr_CO2_exp,
             #xerr=,
             #yerr=,
             color=Color_for_Plot_Yazaydin_Snurr_exp,
             marker=Yazaydin_Snurr_marker_exp,
             linestyle='none',
             markersize=PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Yazaydin_Snurr_exp,
             fillstyle='full',
             fmt=error_bar_fmt,
             ecolor=Color_for_Plot_Yazaydin_Snurr_exp,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

plt.errorbar(P_bar_mean_Yazaydin_Snurr_CO2_sim,
             adsorpbed_mean_Yazaydin_Snurr_CO2_sim,
             #xerr=,
             #yerr=,
             color=Color_for_Plot_Yazaydin_Snurr_sim,
             marker=Yazaydin_Snurr_marker_sim,
             linestyle='none',
             markersize=PointSizes,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label_Yazaydin_Snurr_sim,
             fillstyle='none',
             fmt=error_bar_fmt,
             ecolor=Color_for_Plot_Yazaydin_Snurr_sim,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )


print(f"P_bar_mean_CO2 = {P_bar_mean_CO2}")
print(f"adsorpbed_mean_CO2 = {adsorpbed_mean_CO2}")
plt.errorbar(P_bar_mean_CO2,
             adsorpbed_mean_CO2,
             #xerr=,
             #yerr=adsorpbed_std_CO2,
             color=Color_for_Plot_Data,
             marker=marker,
             linestyle='none',
             markersize=PointSizes+4,
             markeredgewidth=MarkerWidth,
             linewidth=ConnectionLineSizes,
             label=File_label,
             fillstyle='none',
             fmt=error_bar_fmt,
             ecolor=Color_for_Plot_Data,
             elinewidth=error_bar_elinewidth,
             capsize=error_bar_capsize
             )

major_xticks = np.arange(pressure_bar_min,
                         pressure_bar_max+0.001,
                         pressure_bar_ticks_major)
major_yticks = np.arange(adsorbed_molecules_per_unit_cell_min,
                         adsorbed_molecules_per_unit_cell_max+0.001,
                         adsorbed_molecules_per_unit_cell_major)

minor_xticks = np.arange(pressure_bar_min,
                         pressure_bar_max+0.001,
                         pressure_bar_ticks_minor )
minor_yticks = np.arange(adsorbed_molecules_per_unit_cell_min,
                         adsorbed_molecules_per_unit_cell_max+0.001,
                         adsorbed_molecules_per_unit_cell_minor)


plt.tight_layout()  # centers layout nice for final paper


pad_whitepace_for_numbering = 10
plt.tick_params(axis='both', which='major', length=12, width=2, direction='in',
                labelsize=axis_number_font_size, top=True, right=True,
                pad=pad_whitepace_for_numbering)
plt.tick_params(axis='both', which='minor', length=4, width=1, direction='in',
                labelsize=axis_number_font_size, top=True, right=True,
                pad=pad_whitepace_for_numbering)

plt.xlim(pressure_bar_min, pressure_bar_max+0.0001)  # set plot range on x axis
plt.ylim(adsorbed_molecules_per_unit_cell_min, adsorbed_molecules_per_unit_cell_max+0.0001)  # set plot range on x axis


ax1.set_xscale("log", nonpositive='clip')
ax1.set_yscale("log", nonpositive='clip')

plt.xticks(fontsize=axis_number_font_size, rotation=0)
plt.yticks(fontsize=axis_number_font_size, rotation=0)

plt.gcf().subplots_adjust(left=0.20, bottom=0.2, right=0.95, top=0.94, wspace=None, hspace=None) # moves plot  so x label not cutoff

plt.show()
fig1.savefig(f"adsorbed_moleules_per_unit_cell_file_saving_name_CO2_{'log_log'}")

#****************************************
#Plot Number 1  (ln10 CO2 molecule / unit cell vs log10 pressure) (end)
#****************************************


