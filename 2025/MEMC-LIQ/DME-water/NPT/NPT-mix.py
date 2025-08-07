"""GOMC's setup for signac, signac-flow, signac-dashboard for this study."""
# project.py

import os
import subprocess
import shutil
import flow

import mbuild as mb
import mosdef_gomc.formats.gmso_charmm_writer as mf_charmm
import mosdef_gomc.formats.gmso_gomc_conf_writer as gomc_control
import numpy as np
import signac
import unyt as u
import pandas as pd
import math
from scipy import stats
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment


class Project(FlowProject):
    """Subclass of FlowProject to provide custom methods and attributes."""

    def __init__(self):
        super().__init__()

class Grid(DefaultSlurmEnvironment):  # Grid(StandardEnvironment):
    """Subclass of DefaultSlurmEnvironment for WSU's Grid cluster."""
    
    #uncomment for Grid
    #hostname_pattern = r".*\.grid\.wayne\.edu"
    #template = "grid.sh"
    template = "local.sh"




# ******************************************************
# users typical variables, but not all (start)
# ******************************************************
# set binary path to gomc binary files (the bin folder).
# If the gomc binary files are callable directly from the terminal without a path,
# please just enter and empty string (i.e., "" or '')

# Enter the GOMC binary path here (MANDATORY INFORMAION)
#gomc_binary_path = "/home6/ai8111/bin"
gomc_binary_path = "~/GOMC/v012025/GOMC/bin"


# number of simulation steps
gomc_steps_nvt_equilibration =5000000 # pre-equilibrate system without swap moves.
gomc_steps_equilibration = 100000000 #  set value for paper = 60 * 10**6
gomc_steps_production = 100000000 # set value for paper = 60 * 10**6
console_output_freq = 100000 # Monte Carlo Steps between console output
pressure_calc_freq = 1000 # Monte Carlo Steps for pressure calculation
block_ave_output_freq = 10000000 # Monte Carlo Steps between console output
coordinate_output_freq = 1000000 # # set value for paper = 50 * 10**3
EqSteps = 1000000 # MCS for equilibration
AdjSteps =100000 #MCS for adjusting max displacement, rotation, volume, etc.

# number of simulation steps
#gomc_steps_equilb_design_ensemble = 60 * 10**6 #  set value for paper = 60 * 10**6
#gomc_steps_production = 60 * 10**6 # set value for paper = 60 * 10**6

gomc_output_data_every_X_steps = 50 * 10**3 # # set value for paper = 50 * 10**3

# force field (FF) file for all simulations in that job
# Note: do not add extensions
gomc_ff_filename_str = "TraPPE_FF"

# NVT preequilibration to help stabilize GEMC simulations
gomc_nvt_equilb_control_file_name_str = "NPT_nvt"
gomc_nvt_equilb_output_name_str="DME-WAT_nvt"

# initial mosdef structure and coordinates
# Note: do not add extensions
mosdef_structure_box_0_name_str = "initial_box_0"
mosdef_structure_box_1_name_str = "initial_box_1"

# The equilb using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
gomc_equilb_control_file_name_str = "NPT_equil"
gomc_equilb_output_name_str="DME-WAT_equil"

# The production run using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
gomc_production_control_file_name_str = "NPT_prod"
gomc_production_output_name_str="DME-WAT_prod"

# The equilb using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
#gomc_equilb_design_ensemble_control_file_name_str = "gomc_equilb_design_ensemble"

# The production run using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
#gomc_production_control_file_name_str = "gomc_production_run"

# Analysis (each replicates averages):
# Output text (txt) file names for each replicates averages
# directly put in each replicate folder (.txt, .dat, etc)
output_replicate_txt_file_name_liq = "avg_data_box_liq.txt"


# Analysis (averages and std. devs. of  # all the replcates): 
# Output text (txt) file names for the averages and std. devs. of all the replcates, 
# including the extention (.txt, .dat, etc)
output_avg_std_of_replicates_txt_file_name_liq = "avg_over_replicates_box_liq.txt"

# Analysis (Critical and boiling point values):
# Output text (txt) file name for the Critical and Boiling point values of the system using the averages
# and std. devs. all the replcates, including the extention (.txt, .dat, etc)
output_critical_data_replicate_txt_file_name = "critical_points_all_replicates.txt"
output_critical_data_avg_std_of_replicates_txt_file_name = "critical_point_avg_over_replicates.txt"

output_boiling_data_replicate_txt_file_name = "boiling_point_all_replicates.txt"
output_boiling_data_avg_std_of_replicates_txt_file_name = "boiling_point_avg_over_replicates.txt"


walltime_mosdef_hr = 24
walltime_gomc_equilbrium_hr = 200
walltime_gomc_production_hr = 200
walltime_gomc_analysis_hr = 4
memory_needed = 1

# ******************************************************
# users typical variables, but not all (end)
# ******************************************************


# ******************************************************
# signac and GOMC-MOSDEF code (start)
# ******************************************************

# ******************************************************
# ******************************************************
# create some initial variable to be store in each jobs
# directory in an additional json file, and test
# to see if they are written (start).
# ******************************************************
# ******************************************************

# set the default directory
project_directory_path = str(os.getcwd())
print("project_directory_path = " +str(project_directory_path))


@Project.label
def part_1a_initial_data_input_to_json(job):
    """Check that the initial job data is written to the json files."""
    data_written_bool = False
    if job.isfile(f"{'signac_job_document.json'}"):
        data_written_bool = True

    return data_written_bool


@Project.post(part_1a_initial_data_input_to_json)
@Project.operation(directives=
    {
        "np": 1,
        "ngpu": 0,
        "memory": memory_needed,
        "walltime": walltime_mosdef_hr,
    }, with_job=True
)
def initial_parameters(job):
    """Set the initial job parameters into the jobs doc json file."""
    # select

    # set free energy data in doc
    # Free energy calcs
    # lamda generator

    # list replica seed numbers
    replica_no_to_seed_dict = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
    }

    job.doc.replica_number_int = replica_no_to_seed_dict.get(
        int(job.sp.replica_number_int)
    )

    # gomc core and CPU or GPU
    job.doc.gomc_ncpu = 4  # 4 is optimal for water
    job.doc.gomc_ngpu = 1

    # get the gomc binary paths
    if job.doc.gomc_ngpu == 0:
        job.doc.gomc_cpu_or_gpu = "CPU"

    elif job.doc.gomc_ngpu == 1:
        job.doc.gomc_cpu_or_gpu = "GPU"

    else:
        raise ValueError(
            "The GOMC CPU and GPU can not be determined as force field (FF) is not available in the selection, "
            "or GPU selection is is not 0 or 1."
        )

    # Set job executable
    job.doc.gomc_equilb_design_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NPT"
    job.doc.gomc_production_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NPT"


# ******************************************************
# ******************************************************
# create some initial variable to be store in each jobs
# directory in an additional json file, and test
# to see if they are written (end).
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# functions for selecting/grouping/aggregating in different ways (start)
# ******************************************************
# ******************************************************

def statepoint_without_replica(job):
    keys = sorted(tuple(i for i in job.sp.keys() if i not in {"replica_number_int"}))
    return [(key, job.sp[key]) for key in keys]

def statepoint_without_temperature(job):
    keys = sorted(tuple(i for i in job.sp.keys() if i not in {"production_temperature_K"}))
    return [(key, job.sp[key]) for key in keys]

# ******************************************************
# ******************************************************
# functions for selecting/grouping/aggregating in different ways (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC psf, pdb, and force field (FF) files were written (start)
# ******************************************************
# ******************************************************

# check if GOMC-MOSDEF wrote the gomc files
@Project.label
def mosdef_input_written(job):
    """Check that the mosdef files (psf, pdb, and force field (FF) files) are written ."""
    file_written_bool = False
    if (
        job.isfile(f"{gomc_ff_filename_str}.inp")
        and job.isfile(
            f"{mosdef_structure_box_0_name_str}.psf"
        )
        and job.isfile(
            f"{mosdef_structure_box_0_name_str}.pdb"
        )
    ):
        file_written_bool = True

    return file_written_bool


# ******************************************************
# ******************************************************
# check if GOMC psf, pdb, and FF files were written (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC control file was written (start)
# ******************************************************
# ******************************************************
# function for checking if the GOMC control file is written
def gomc_control_file_written(job, control_filename_str):
    """General check that the gomc control files are written."""
    file_written_bool = False
    control_file = f"{control_filename_str}.conf"

    if job.isfile(control_file):
        with open(job.fn(control_file), "r") as fp:
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "OutputName" in line:
                    split_move_line = line.split()
                    if split_move_line[0] == "OutputName":
                        file_written_bool = True

    return file_written_bool

# checking if the GOMC control file is written for the nvt equilb run with the selected ensemble
@Project.label
def part_2a_gomc_nvt_equilb_design_ensemble_control_file_written(job):
    """General check that the gomc_equilb_design_ensemble (run temperature) gomc control file is written."""
    return gomc_control_file_written(job, gomc_nvt_equilb_control_file_name_str)

# checking if the GOMC control file is written for the equilb run with the selected ensemble
@Project.label
def part_2b_gomc_equilb_design_ensemble_control_file_written(job):
    """General check that the gomc_equilb_design_ensemble (run temperature) gomc control file is written."""
    return gomc_control_file_written(job, gomc_equilb_control_file_name_str)


# checking if the GOMC control file is written for the production run
@Project.label
def part_2c_gomc_production_control_file_written(job):
    """General check that the gomc_production_control_file (run temperature) is written."""
    return gomc_control_file_written(job, gomc_production_control_file_name_str)

# ******************************************************
# ******************************************************
# check if GOMC control file was written (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC simulations started (start)
# ******************************************************
# ******************************************************
# function for checking if GOMC simulations are started
def gomc_simulation_started(job, control_filename_str):
    """General check to see if the gomc simulation is started."""
    output_started_bool = False
    if job.isfile("out_{}.dat".format(control_filename_str)):
        output_started_bool = True

    return output_started_bool

@Project.label
def part_3a_output_gomc_nvt_equilb_design_ensemble_started(job):
    """Check to see if the gomc_equilb_design_ensemble (set temperature) simulation is started."""
    return gomc_simulation_started(job, gomc_nvt_equilb_control_file_name_str)

# check if equilb_with design ensemble GOMC run is started by seeing if the GOMC consol file and the merged psf exist
@Project.label
def part_3b_output_gomc_equilb_design_ensemble_started(job):
    """Check to see if the gomc_equilb_design_ensemble (set temperature) simulation is started."""
    return gomc_simulation_started(job, gomc_equilb_control_file_name_str)

# check if production GOMC run is started by seeing if the GOMC consol file and the merged psf exist
@Project.label
def part_3c_output_gomc_production_run_started(job):
    """Check to see if the gomc production run (set temperature) simulation is started."""
    return gomc_simulation_started(job, gomc_production_control_file_name_str)


# ******************************************************
# check if GOMC simulations started (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC simulation are completed properly (start)
# ******************************************************
# ******************************************************
# function for checking if GOMC simulations are completed properly
def gomc_sim_completed_properly(job, control_filename_str):
    """General check to see if the gomc simulation was completed properly."""
    job_run_properly_bool = False
    output_log_file = "out_{}.dat".format(control_filename_str)
    if job.isfile(output_log_file):
        # with open(f"workspace/{job.id}/{output_log_file}", "r") as fp:
        #print(f"job.id = {job.id}")
        with open(job.fn(output_log_file), "r") as fp:
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "Completed" in line:
                    job_run_properly_bool = True
              #  if "Move" in line:
              #      split_move_line = line.split()
              #      if (
              #          split_move_line[0] == "Move"
              #          and split_move_line[1] == "Type"
              #          and split_move_line[2] == "Mol."
              #          and split_move_line[3] == "Kind"
              #      ):
              #          job_run_properly_bool = True
    else:
        job_run_properly_bool = False

    return job_run_properly_bool

# check if equilb selected ensemble GOMC run completed by checking the end of the GOMC consol file
# check if equilb selected ensemble GOMC run completed by checking the end of the GOMC consol file
@Project.label
def part_4a_job_gomc_nvt_equilb_design_ensemble_completed_properly(job):
    """Check to see if the gomc_equilb_design_ensemble (set temperature) simulation was completed properly."""
    return gomc_sim_completed_properly(job, gomc_nvt_equilb_control_file_name_str)

@Project.label
def part_4b_job_gomc_equilb_design_ensemble_completed_properly(job):
    """Check to see if the gomc_equilb_design_ensemble (set temperature) simulation was completed properly."""
    return gomc_sim_completed_properly(job, gomc_equilb_control_file_name_str)

# check if production GOMC run completed by checking the end of the GOMC consol file
@Project.label
def part_4c_job_production_run_completed_properly(job):
    """Check to see if the gomc production run (set temperature) simulation was completed properly."""
    return gomc_sim_completed_properly(job, gomc_production_control_file_name_str)



# check if analysis is done for the individual replicates wrote the gomc files
@Project.label
def part_5a_analysis_individual_simulation_averages_completed(job):
    """Check that the individual simulation averages files are written ."""
    file_written_bool = False
    if (
        job.isfile(
            f"{output_replicate_txt_file_name_liq}"
        )
    ):
        file_written_bool = True

    return file_written_bool


# check if analysis for averages of all the replicates is completed
@Project.label
def part_5b_analysis_replica_averages_completed(*jobs):
    """Check that the individual simulation averages files are written ."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_avg_std_of_replicates_txt_file_name_liq}"
            )
            
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return all_file_written_bool_pass

# check if analysis for critical points is completed
@Project.label
def part_5c_analysis_critical_and_boiling_points_replicate_data_completed(*jobs):
    """Check that the critical and boiling point replicate file is written ."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_critical_data_replicate_txt_file_name}"
            ) \
                and
            job.isfile(
                f"../../analysis/{output_boiling_data_replicate_txt_file_name}"
            )
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return file_written_bool

# check if analysis for critical points is completed
@Project.label
def part_5d_analysis_critical_and_boiling_points_avg_std_data_completed(*jobs):
    """Check that the avg and std dev critical and boiling point data file is written ."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_critical_data_avg_std_of_replicates_txt_file_name}"
            ) \
                and
                job.isfile(
                    f"../../analysis/{output_boiling_data_avg_std_of_replicates_txt_file_name}"
                )
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return file_written_bool


# ******************************************************
# ******************************************************
# check if GOMC simulation are completed properly (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# build system, with option to write the force field (force field (FF)), pdb, psf files.
# Note: this is needed to write GOMC control file, even if a restart (start)
# ******************************************************
# build system
def build_charmm(job, write_files=True):
    """Build the Charmm object and potentially write the pdb, psd, and force field (FF) files."""
    print("#**********************")
    print("Started: GOMC Charmm Object")
    print("#**********************")

    box_0_temp_to_length_ang_dict = { 190: 69, 
                                       180: 69,
                                       170: 69,
                                       160: 69,
                                       150: 69,
                                       140: 69,
                                       130: 69,
                                       120: 69,
                                       110: 69,
                                       106.7:69,
                                       100: 69,
                                       96: 69,
                                       95: 69,
                                       94: 69,
                                       93: 69,
                                       92: 69,
                                       91: 69,
                                       90: 69,
                                       89: 69,
                                       88: 69,
                                       87: 69,
                                       86: 69,
                                       85: 69,
                                     }

    box_1_temp_to_length_ang_dict = { 190:71, 
                                      180:71, 
                                      170:71,
                                      160:71,
                                      150:71,
                                      140:71, 
                                      130:71, 
                                      120:71, 
                                      110:71,
                                      106.7:71, 
                                      100:71, 
                                      96:71, 
                                      95:71, 
                                      94:71, 
                                      93:71, 
                                      92:71, 
                                      91:71, 
                                      90:71, 
                                      89:71, 
                                      88:71, 
                                      87:71, 
                                      86:71, 
                                      85:71, 
                                     }

    box_0_temp_to_mol_frac_dict = {  190:0.5,
                                     180:0.5,
                                     170:0.5,
                                     160:0.5,
                                     150:0.5,   
                                     140:0.5,   
                                     130:0.5,   
                                     120:0.5,   
                                     110:0.5,   
                                     100:0.5,   
                                     96:0.5,   
                                     95:0.5,   
                                     94:0.5,   
                                     93:0.5,   
                                     92:0.5,   
                                     91:0.5,   
                                     90:0.5,   
                                     89:0.5,   
                                     88:0.5,   
                                     87:0.5,   
                                     86:0.5,   
                                     85:0.5,   
                                     }

    box_1_temp_to_mol_frac_dict = { 190: 0.5,
                                    180: 0.5,
                                    170: 0.5,
                                   160: 0.5,
                                   150: 0.5,
                                   140: 0.5,
                                   130: 0.5,
                                   120: 0.5,
                                   110: 0.5,
                                   100: 0.5,
                                   96: 0.5,
                                   95: 0.5,
                                   94: 0.5,
                                   93: 0.5,
                                   92: 0.5,
                                   91: 0.5,
                                   90: 0.5,
                                   89: 0.5,
                                   88: 0.5,
                                   87: 0.5,
                                   86: 0.5,
                                   85: 0.5,
                                     }

    total_molecules_liquid = 5000

    water_forcefield = '../../SPCE_GMSO.xml'
    forcefield = '../../trappe-mie.xml'

    molecule_A=mb.load('../../DME.mol2')
    molecule_A.name = 'DME'
    molecule_B=mb.load('../../SPCE.mol2')
    molecule_B.name = 'SPCE'

    forcefield_files = {molecule_A.name : forcefield, molecule_B.name : water_forcefield}
    molecule_type_list = [molecule_A,molecule_B]

    #molecule_mol_fraction_list = [mol_fraction_molecule_A,mol_fraction_molecule_B]
    fixed_bonds_list = [molecule_A.name,molecule_B.name]
    fixed_bonds_angles_list = [molecule_B.name]

    residues_list = [molecule_A.name,molecule_B.name]

    
    bead_to_atom_name_dict = {'_FP':'FP','_NE': 'NE','_AR':'AR','_CH4': 'C', '_CH3': 'C', '_CH2': 'C', '_CH': 'C', '_HC': 'C','_CF4':'C', '_CF3':'C','_CF2':'C'}

    #print('total_molecules_liquid = ' + str(total_molecules_liquid))
    #print('total_molecules_vapor = ' + str(total_molecules_vapor))

    box_0_box_size_ang = 100.0

    #box_0_box_size_ang = box_0_temp_to_length_ang_dict[job.sp.production_temperature_K]
    #box_1_box_size_ang = box_1_temp_to_length_ang_dict[job.sp.production_temperature_K]

    box_0_molecules=[int(total_molecules_liquid*job.sp.production_composition),
    int(total_molecules_liquid*(1-job.sp.production_composition))]

    print('Running: liquid phase box packing')
    box_liq = mb.fill_box(compound=molecule_type_list,
                          n_compounds=box_0_molecules,seed=job.doc.replica_number_int,overlap=0.25,
                          box=[box_0_box_size_ang/10, box_0_box_size_ang/10, box_0_box_size_ang/10]
                          )
    print('Completed: liquid phase box packing')

    print('Running: GOMC FF file, and the psf and pdb files')
    # gms_match_ff_by = 'group" needed to set the methane residue name
    gomc_charmm = mf_charmm.Charmm(
        box_liq,
        mosdef_structure_box_0_name_str,
        ff_filename=gomc_ff_filename_str,
        forcefield_selection=forcefield_files,
        residues=residues_list,
        bead_to_atom_name_dict=bead_to_atom_name_dict,
        gomc_fix_bonds=fixed_bonds_list,
        gomc_fix_bonds_angles=fixed_bonds_angles_list,
        gmso_match_ff_by='group',
    )

    if write_files == True:
        gomc_charmm.write_inp()

        gomc_charmm.write_psf()

        gomc_charmm.write_pdb()

# kludge for NBFIX
    NBFIX_out = 'CH40 CF40 143.71778216 3.94 12\n'
   
    file=open(gomc_ff_filename_str+'.inp','r')
    fw=open("temp_ff","w+")
    for line in file:
        if not "END" in line:
            fw.write(line)
        if "END" in line:
            line=line.replace(line,"NBFIX_MIE \n")
            fw.write(line)
            fw.write(NBFIX_out+'\n')
            fw.write("END")
    fw.close()
    shutil.move('temp_ff',gomc_ff_filename_str+'.inp')

    print('Completed: GOMC FF file, and the psf and pdb files')

    return gomc_charmm


# ******************************************************
# ******************************************************
# build system, with option to write the force field (FF), pdb, psf files.
# Note: this is needed to write GOMC control file, even if a restart (end)
# ******************************************************


# ******************************************************
# ******************************************************
# Creating GOMC files (pdb, psf, force field (FF), and gomc control files (start)
# ******************************************************
# ******************************************************
@Project.pre(part_1a_initial_data_input_to_json)
@Project.post(part_2a_gomc_nvt_equilb_design_ensemble_control_file_written)
@Project.post(part_2b_gomc_equilb_design_ensemble_control_file_written)
@Project.post(part_2c_gomc_production_control_file_written)
@Project.post(mosdef_input_written)
@Project.operation(directives=
    {
        "np": 1,
        "ngpu": 0,
        "memory": memory_needed,
        "walltime": walltime_mosdef_hr,
    }, with_job=True
)
def build_psf_pdb_ff_gomc_conf(job):
    """Build the Charmm object and write the pdb, psd, and force field (FF) files for all the simulations in the workspace."""
    gomc_charmm_object_with_files = build_charmm(job, write_files=True)

    # ******************************************************
    # common variables (start)
    # ******************************************************
    # variables from signac workspace
    box_0_temp_to_rcutcoulomb_dict = {300:12,
                                     350: 12,
                                     400: 12,
                                     450: 12,
                                     500: 12,
                                     11.112: 12,
                                     575: 12,
                                     600: 12,
                                     615: 10,
                                     625: 10,
                                     }
    
    box_1_temp_to_rcutcoulomb_dict = {300:100,
                                     350: 50,
                                     400: 30,
                                     450: 20,
                                     500: 20,
                                     11.112: 12,
                                     575: 12,
                                     600: 10,
                                     615: 10,
                                     625: 10,
                                     }
    
    production_temperature_K = job.sp.production_temperature_K * u.K
    production_pressure_bar = job.sp.production_pressure_bar * u.bar
    seed_no = job.doc.replica_number_int

    # cutoff and tail correction
    #TIP3P-ML rcut = 7.5 A
    Rcut_ang = 10.0 * u.angstrom
    Rcut_low_ang = 1.1 * u.angstrom
    LRC = True
    Ewald_bool = True
    Electrostatic_bool = False
    Ewald_tol=0.001
    Exclude = "1-4"
    #using default values for RcutCoulomb (same as Rcut for LJ interactions)
    #RcutCoulomb_box_0 = box_0_press_to_rcutcoulomb_dict[job.sp.production_pressure_bar]*u.angstrom
    #RcutCoulomb_box_1 = box_1_press_to_rcutcoulomb_dict[job.sp.production_pressure_bar]*u.angstrom
    #RcutCoulomb_box_0=10.0 * u.angstrom
    #RcutCoulomb_box_1=10.0 * u.angstrom

    # output all data and calc frequecy
    output_true_list_input = [
        True,
        int(gomc_output_data_every_X_steps),
    ]
    output_false_list_input = [
        False,
        int(gomc_output_data_every_X_steps),
    ]

    # ******************************************************
    # common variables (end))
    # ******************************************************
    print("#**********************")
    print("Started: NVT Pre-equilibration GEMC-NVT GOMC control file writing")
    print("#**********************")
    output_file_prefix = gomc_nvt_equilb_output_name_str
    starting_control_file_name_str = gomc_charmm_object_with_files

# calc MC steps for gomc equilb
    MC_steps = int(gomc_steps_nvt_equilibration)
    Expert_bool = True
    # MC move ratios
    DisFreq = 0.4
    RotFreq = 0.4
    VolFreq = 0.0
    MultiParticleFreq=0.00
    RegrowthFreq = 0.00
    IntraSwapFreq = 0.00
    IntraMEMC_2Freq = 0.20
    MEMC_2LiqFreq = 0.0
    CrankShaftFreq = 0.00
    SwapFreq = 0.0
    MEMC_2Freq = 0.0
    memc_move_1 = [1, "DME", ["C1","O1"], "SPCE", ["H1","O1"]]
    memc_all_moves = [memc_move_1]
    Exchange_volume = [3.0, 3.0, 3.0]

    gomc_control.write_gomc_control_file(
        starting_control_file_name_str,
        gomc_nvt_equilb_control_file_name_str,
        'NPT',
        MC_steps,
        production_temperature_K,
        ff_psf_pdb_file_directory=None,
        check_input_files_exist=False,
        Parameters=None,
        Restart=False,
        Checkpoint=False,
        ExpertMode=Expert_bool,
        Coordinates_box_0=None,
        Structure_box_0=None,
        binCoordinates_box_0=None,
        extendedSystem_box_0=None,
        binVelocities_box_0=None,
        Coordinates_box_1=None,
        Structure_box_1=None,
        binCoordinates_box_1=None,
        extendedSystem_box_1=None,
        binVelocities_box_1=None,
        input_variables_dict={
            "PRNG": seed_no,
            "Pressure": production_pressure_bar,
            "Potential": "VDW",
            "LRC": LRC,
            "Rcut": Rcut_ang,
            "RcutLow": Rcut_low_ang,
        #    "RcutCoulomb_box_0":RcutCoulomb_box_0,
        #    "RcutCoulomb_box_1":RcutCoulomb_box_1,
            "Ewald": Ewald_bool,
            "ElectroStatic": Electrostatic_bool,
            "Tolerance": Ewald_tol,
            "VDWGeometricSigma": False,
            "Exclude": Exclude,
            "DisFreq": DisFreq,
            "VolFreq": VolFreq,
            "RotFreq": RotFreq,
            "MultiParticleFreq": MultiParticleFreq,
            "RegrowthFreq": RegrowthFreq,
            "IntraSwapFreq": IntraSwapFreq,
            "IntraMEMC-2Freq": IntraMEMC_2Freq,
            "CrankShaftFreq": CrankShaftFreq,
            "SwapFreq": SwapFreq,
            "MEMC-2Freq": MEMC_2Freq,
            "MEMC-2-LiqFreq": MEMC_2LiqFreq,
            "ExchangeVolumeDim": Exchange_volume,
            "MEMC_DataInput": memc_all_moves,
            "OutputName": output_file_prefix,
            "EqSteps": EqSteps,
            "AdjSteps":AdjSteps,
            "PressureCalc": [False, pressure_calc_freq],
            "RestartFreq": [True, coordinate_output_freq],
            "CheckpointFreq": [True, coordinate_output_freq],
            "DCDFreq": [True, coordinate_output_freq],
            "ConsoleFreq": [True, console_output_freq],
            "BlockAverageFreq":[True, block_ave_output_freq],
            "HistogramFreq": output_false_list_input,
            "CoordinatesFreq": output_false_list_input,
            "CBMC_First": 12,
            "CBMC_Nth": 10,
            "CBMC_Ang": 50,
            "CBMC_Dih": 50,
        },
    )
    print("#**********************")
    print("Completed: pre-equilb GEMC-NVT GOMC control file writing")
    print("#**********************")

    # ******************************************************
    # equilb selected_ensemble, if NVT -> NPT - GOMC control file writing  (start)
    # Note: the control files are written for the max number of gomc_equilb_design_ensemble runs
    # so the Charmm object only needs created 1 time.
    # ******************************************************
    print("#**********************")
    print("Started: equilb GEMC-NVT GOMC control file writing")
    print("#**********************")
      # Set Monte Carlo steps for equilibration MC steps for gomc
    MC_steps = int(gomc_steps_equilibration)
    
    # MC move ratios
    DisFreq = 0.45
    RotFreq = 0.44
    VolFreq = 0.01
    MultiParticleFreq=0.00
    RegrowthFreq = 0.00
    IntraSwapFreq = 0.00
    IntraMEMC_2Freq = 0.10
    MEMC_2LiqFreq = 0.0
    CrankShaftFreq = 0.00
    SwapFreq = 0.00
    MEMC_2Freq = 0.0
    memc_move_1 = [1, "DME", ["C1","O1"], "SPCE", ["H1","O1"]]
    Exchange_volume = [4.0, 4.0, 4.0]
    memc_all_moves = [memc_move_1]

    output_file_prefix = gomc_equilb_output_name_str
    starting_control_file_name_str = gomc_charmm_object_with_files
    restart_control_file_name_str = gomc_nvt_equilb_output_name_str
# setup file names for the second stage of the equilibration
    Coordinates_box_0 = "{}_BOX_0_restart.pdb".format(
        restart_control_file_name_str
    )   
    Structure_box_0 = "{}_BOX_0_restart.psf".format(
        restart_control_file_name_str
    )
    binCoordinates_box_0 = "{}_BOX_0_restart.coor".format(
        restart_control_file_name_str
    )
    extendedSystem_box_0 = "{}_BOX_0_restart.xsc".format(
        restart_control_file_name_str
    )
    Coordinates_box_1 = "{}_BOX_1_restart.pdb".format(
        restart_control_file_name_str
    )
    Structure_box_1 = "{}_BOX_1_restart.psf".format(
        restart_control_file_name_str
    )
    binCoordinates_box_1 = "{}_BOX_1_restart.coor".format(
        restart_control_file_name_str
    )
    extendedSystem_box_1 = "{}_BOX_1_restart.xsc".format(
        restart_control_file_name_str
    )
   
    gomc_control.write_gomc_control_file(
        starting_control_file_name_str,
        gomc_equilb_control_file_name_str,
        'NPT',
        MC_steps,
        production_temperature_K,
        ff_psf_pdb_file_directory=None,
        check_input_files_exist=False,
        Parameters=None,
        Restart=True,
        Checkpoint=False,
        ExpertMode=False,
        Coordinates_box_0=Coordinates_box_0,
        Structure_box_0=Structure_box_0,
        binCoordinates_box_0=binCoordinates_box_0,
        extendedSystem_box_0=extendedSystem_box_0,
        Coordinates_box_1=Coordinates_box_1,
        Structure_box_1=Structure_box_1,
        binCoordinates_box_1=binCoordinates_box_1,
        extendedSystem_box_1=extendedSystem_box_1,
        binVelocities_box_0=None,
        binVelocities_box_1=None,
        input_variables_dict={
            "PRNG": seed_no,
            "Pressure": production_pressure_bar,
            "Potential": "VDW",
            "LRC": LRC,
            "Rcut": Rcut_ang,
            "RcutLow": Rcut_low_ang,
        #    "RcutCoulomb_box_0":RcutCoulomb_box_0,
        #    "RcutCoulomb_box_1":RcutCoulomb_box_1,
            "Ewald": Ewald_bool,
            "ElectroStatic": Electrostatic_bool,
            "Tolerance": Ewald_tol,
            "VDWGeometricSigma": False,
            "Exclude": Exclude,
            "DisFreq": DisFreq,
            "VolFreq": VolFreq,
            "RotFreq": RotFreq,
            "MultiParticleFreq": MultiParticleFreq,
            "RegrowthFreq": RegrowthFreq,
            "IntraSwapFreq": IntraSwapFreq,
            "IntraMEMC-2Freq": IntraMEMC_2Freq,
            "CrankShaftFreq": CrankShaftFreq,
            "SwapFreq": SwapFreq,
            "MEMC-2Freq": MEMC_2Freq,
            "MEMC-2-LiqFreq": MEMC_2LiqFreq,
            "ExchangeVolumeDim": Exchange_volume,
            "MEMC_DataInput": memc_all_moves,
            "OutputName": output_file_prefix,
            "EqSteps": EqSteps,
            "AdjSteps":AdjSteps,
            "PressureCalc": [False, pressure_calc_freq],
            "RestartFreq": [True, coordinate_output_freq],
            "CheckpointFreq": [True, coordinate_output_freq],
            "DCDFreq": [True, coordinate_output_freq],
            "ConsoleFreq": [True, console_output_freq],
            "BlockAverageFreq":[True, block_ave_output_freq],
            "HistogramFreq": output_false_list_input,
            "CoordinatesFreq": output_false_list_input,
            "CBMC_First": 12,
            "CBMC_Nth": 10,
            "CBMC_Ang": 50,
            "CBMC_Dih": 50,
        },
    )
    print("#**********************")
    print("Completed: equilb GEMC-NPT GOMC control file written")
    print("#**********************")

    # ******************************************************
    # equilb selected_ensemble, if NVT -> NPT - GOMC control file writing  (end)
    # Note: the control files are written for the max number of gomc_equilb_design_ensemble runs
    # so the Charmm object only needs created 1 time.
    # ******************************************************

    # ******************************************************
    # production NPT or GEMC-NVT - GOMC control file writing  (start)
    # ******************************************************

    print("#**********************")
    print("Started: production GEMC-NPT GOMC control file writing")
    print("#**********************")

    output_file_prefix = gomc_production_output_name_str
    restart_control_file_name_str = gomc_equilb_output_name_str

    # calc MC steps
    MC_steps = int(gomc_steps_production)

    Coordinates_box_0 = "{}_BOX_0_restart.pdb".format(
        restart_control_file_name_str
    )
    Structure_box_0 = "{}_BOX_0_restart.psf".format(
        restart_control_file_name_str
    )
    binCoordinates_box_0 = "{}_BOX_0_restart.coor".format(
        restart_control_file_name_str
    )
    extendedSystem_box_0 = "{}_BOX_0_restart.xsc".format(
        restart_control_file_name_str
    )

    Coordinates_box_1 = "{}_BOX_1_restart.pdb".format(
        restart_control_file_name_str
    )
    Structure_box_1 = "{}_BOX_1_restart.psf".format(
        restart_control_file_name_str
    )
    binCoordinates_box_1 = "{}_BOX_1_restart.coor".format(
        restart_control_file_name_str
    )
    extendedSystem_box_1 = "{}_BOX_1_restart.xsc".format(
        restart_control_file_name_str
    )

    gomc_control.write_gomc_control_file(
        gomc_charmm_object_with_files,
        gomc_production_control_file_name_str,
        "NPT",
        MC_steps,
        production_temperature_K,
        ff_psf_pdb_file_directory=None,
        check_input_files_exist=False,
        Parameters=None,
        Restart=True,
        Checkpoint=False,
        ExpertMode=False,
        Coordinates_box_0=Coordinates_box_0,
        Structure_box_0=Structure_box_0,
        binCoordinates_box_0=binCoordinates_box_0,
        extendedSystem_box_0=extendedSystem_box_0,
        Coordinates_box_1=Coordinates_box_1,
        Structure_box_1=Structure_box_1,
        binCoordinates_box_1=binCoordinates_box_1,
        extendedSystem_box_1=extendedSystem_box_1,
        binVelocities_box_1=None,
        input_variables_dict={
            "PRNG": seed_no,
            "Pressure": production_pressure_bar,
            "Potential": "VDW",
            "LRC": LRC,
            "Rcut": Rcut_ang,
            "RcutLow": Rcut_low_ang,
        #    "RcutCoulomb_box_0":RcutCoulomb_box_0,
        #    "RcutCoulomb_box_1":RcutCoulomb_box_1,
            "Ewald": Ewald_bool,
            "ElectroStatic": Electrostatic_bool,
            "Tolerance": Ewald_tol,
            "VDWGeometricSigma": False,
            "Exclude": Exclude,
            "DisFreq": DisFreq,
            "VolFreq": VolFreq,
            "RotFreq": RotFreq,
            "MultiParticleFreq": MultiParticleFreq,
            "RegrowthFreq": RegrowthFreq,
            "IntraSwapFreq": IntraSwapFreq,
            "IntraMEMC-2Freq": IntraMEMC_2Freq,
            "CrankShaftFreq": CrankShaftFreq,
            "SwapFreq": SwapFreq,
            "MEMC-2Freq": MEMC_2Freq,
            "MEMC-2-LiqFreq": MEMC_2LiqFreq,
            "ExchangeVolumeDim": Exchange_volume,
            "MEMC_DataInput": memc_all_moves,
            "OutputName": output_file_prefix,
            "EqSteps": EqSteps,
            "AdjSteps":AdjSteps,
            "PressureCalc": [False, pressure_calc_freq],
            "RestartFreq": [True, coordinate_output_freq],
            "CheckpointFreq": [True, coordinate_output_freq],
            "DCDFreq": [True, coordinate_output_freq],
            "ConsoleFreq": [True, console_output_freq],
            "BlockAverageFreq":[True, block_ave_output_freq],
            "HistogramFreq": output_false_list_input,
            "CoordinatesFreq": output_false_list_input,            
            "CBMC_First": 12,
            "CBMC_Nth": 10,
            "CBMC_Ang": 50,
            "CBMC_Dih": 50,
        },
    )

    print("#**********************")
    print("Completed: production NPT or GEMC-NVT GOMC control file writing")
    print("#**********************")
    # ******************************************************
    # production NPT or GEMC-NVT - GOMC control file writing  (end)
    # ******************************************************


# ******************************************************
# ******************************************************
# Creating GOMC files (pdb, psf, force field (FF), and gomc control files (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# equilb NPT or GEMC-NVT - starting the GOMC simulation (start)
# ******************************************************
# ******************************************************
@Project.pre(mosdef_input_written)
@Project.pre(part_2a_gomc_nvt_equilb_design_ensemble_control_file_written)
@Project.post(part_4a_job_gomc_nvt_equilb_design_ensemble_completed_properly)
@Project.operation(directives=
    {
        "np": lambda job: job.doc.gomc_ncpu,
        "ngpu": lambda job: job.doc.gomc_ngpu,
        "memory": memory_needed,
        "walltime": walltime_gomc_equilbrium_hr,
    }, with_job=True, cmd=True
)
def run_nvt_equilb_ensemble_gomc_command(job):
    """Run the gomc equilb_ensemble simulation."""
    control_file_name_str = gomc_nvt_equilb_control_file_name_str

    print(f"Running simulation job id {job}")
    run_command = "{}/{} +p{} {}.conf > out_{}.dat".format(
        str(gomc_binary_path),
        str(job.doc.gomc_equilb_design_ensemble_gomc_binary_file),
        str(job.doc.gomc_ncpu),
        str(control_file_name_str),
        str(control_file_name_str),
    )

    print('gomc equilb run_command = ' + str(run_command))

    return run_command


@Project.pre(mosdef_input_written)
@Project.pre(part_2b_gomc_equilb_design_ensemble_control_file_written)
@Project.pre(part_4a_job_gomc_nvt_equilb_design_ensemble_completed_properly)
@Project.post(part_4b_job_gomc_equilb_design_ensemble_completed_properly)
@Project.operation(directives=
    {
        "np": lambda job: job.doc.gomc_ncpu,
        "ngpu": lambda job: job.doc.gomc_ngpu,
        "memory": memory_needed,
        "walltime": walltime_gomc_equilbrium_hr,
    }, with_job=True, cmd=True
)
def run_equilb_ensemble_gomc_command(job):
    """Run the gomc equilb_ensemble simulation."""
    control_file_name_str = gomc_equilb_control_file_name_str

    print(f"Running simulation job id {job}")
    run_command = "{}/{} +p{} {}.conf > out_{}.dat".format(
        str(gomc_binary_path),
        str(job.doc.gomc_equilb_design_ensemble_gomc_binary_file),
        str(job.doc.gomc_ncpu),
        str(control_file_name_str),
        str(control_file_name_str),
    )

    print('gomc equilb run_command = ' + str(run_command))

    return run_command


# ******************************************************
# ******************************************************
# equilb NPT or GEMC-NVT - starting the GOMC simulation (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# production run - starting the GOMC simulation (start)
# ******************************************************
# ******************************************************


@Project.pre(part_2c_gomc_production_control_file_written)
@Project.pre(part_4b_job_gomc_equilb_design_ensemble_completed_properly)
#@Project.post(part_3b_output_gomc_production_run_started)
@Project.post(part_4c_job_production_run_completed_properly)
@Project.operation(directives=
    {
        "np": lambda job: job.doc.gomc_ncpu,
        "ngpu": lambda job: job.doc.gomc_ngpu,
        "memory": memory_needed,
        "walltime": walltime_gomc_production_hr,
    }, with_job=True, cmd=True
)
def run_production_run_gomc_command(job):
    """Run the gomc_production_ensemble simulation."""

    control_file_name_str = gomc_production_control_file_name_str

    print(f"Running simulation job id {job}")
    run_command = "{}/{} +p{} {}.conf > out_{}.dat".format(
        str(gomc_binary_path),
        str(job.doc.gomc_production_ensemble_gomc_binary_file),
        str(job.doc.gomc_ncpu),
        str(control_file_name_str),
        str(control_file_name_str),
    )

    print('gomc production run_command = ' + str(run_command))

    return run_command
# ******************************************************
# ******************************************************
# production run - starting the GOMC simulation (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# data analysis - get the average data from each replicate (start)
# ******************************************************
# ******************************************************

@Project.pre(part_4c_job_production_run_completed_properly)
@Project.post(part_5a_analysis_individual_simulation_averages_completed)
@Project.operation(directives=
     {
         "np": 1,
         "ngpu": 0,
         "memory": memory_needed,
         "walltime": walltime_gomc_analysis_hr,
     }, with_job=True
)
def part_5a_analysis_individual_simulation_averages(job):
    # remove the total averged replicate data and all analysis data after this,
    # as it is no longer valid when adding more simulations
    if os.path.isfile(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_liq}'):
        os.remove(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_liq}')

    # this is set to basically use all values.  However, allows the ability to set if needed
    step_start = 0 * 10 ** 6
    step_finish = 1 * 10 ** 12

    # get the averages from each individual simulation and write the csv's.

    reading_file_box_0 = job.fn(f'Blk_{gomc_production_output_name_str}_BOX_0.dat')
    

    print("HERE")
    output_column_temp_title = 'temp_K'  # column title title for temp
    output_column_no_step_title = 'Step'  # column title title for iter value
    #output_column_no_pressure_title = 'P_bar'  # column title title for PRESSURE
    output_column_total_molecules_title = "No_mol"  # column title title for TOT_MOL
    output_column_Rho_title = 'Rho_kg_per_m_cubed'  # column title title for TOT_DENS
    output_column_box_volume_title = 'V_ang_cubed'  # column title title for VOLUME
    output_column_box_length_if_cubed_title = 'L_m_if_cubed'  # column title title for VOLUME
    #output_column_box_Hv_title = 'Hv_kJ_per_mol'  # column title title for HEAT_VAP
    #output_column_box_Z_title = 'Z'  # column title title for  compressiblity (Z)
#    output_column_box_MF1_title = 'Mol_Frac_F1A'
#    output_column_box_MF2_title = 'Mol_Frac_C1A'


    blk_file_reading_column_no_step_title = '#STEP'  # column title title for iter value
    #blk_file_reading_column_no_pressure_title = 'PRESSURE'  # column title title for PRESSURE
    blk_file_reading_column_total_molecules_title = "TOT_MOL"  # column title title for TOT_MOL
    blk_file_reading_column_Rho_title = 'TOT_DENS'  # column title title for TOT_DENS
    blk_file_reading_column_box_volume_title = 'VOLUME'  # column title title for VOLUME
    #blk_file_reading_column_box_Hv_title = 'HEAT_VAP'  # column title title for HEAT_VAP
    #blk_file_reading_column_box_Z_title = 'COMPRESSIBILITY'  # column title title for compressiblity (Z)
    # in the future, build this out of the residue names
#    blk_file_reading_column_box_mf1_title = 'MOLFRACT_F1A' #column title for MF F1A
#    blk_file_reading_column_box_mf2_title = 'MOLFRACT_C1A' #column title for MF C1A

    # Programmed data
    step_start_string = str(int(step_start))
    step_finish_string = str(int(step_finish))


    # *************************
    # drawing in data from single file and extracting specific rows for the liquid box (start)
    # *************************
    data_box_0 = pd.read_csv(reading_file_box_0, sep='\s+', header=0, na_values='NaN', index_col=False)

    data_box_0 = pd.DataFrame(data_box_0)
    step_no_title_mod = blk_file_reading_column_no_step_title[1:]
    header_list = list(data_box_0.columns)
    header_list[0] = step_no_title_mod
    data_box_0.columns = header_list

    data_box_0 = data_box_0.query(step_start_string + ' <= ' + step_no_title_mod + ' <= ' + step_finish_string)

    iter_no_box_0 = data_box_0.loc[:, step_no_title_mod]
    iter_no_box_0 = list(iter_no_box_0)
    iter_no_box_0 = np.transpose(iter_no_box_0)

#  comment out because we don't have pressure
   # pressure_box_0 = data_box_0.loc[:, blk_file_reading_column_no_pressure_title]
   # pressure_box_0 = list(pressure_box_0)
   # pressure_box_0 = np.transpose(pressure_box_0)
   # pressure_box_0_mean = np.nanmean(pressure_box_0)

    total_molecules_box_0 = data_box_0.loc[:, blk_file_reading_column_total_molecules_title]
    total_molecules_box_0 = list(total_molecules_box_0)
    total_molecules_box_0 = np.transpose(total_molecules_box_0)
    total_molecules_box_0_mean = np.nanmean(total_molecules_box_0)

    Rho_box_0 = data_box_0.loc[:, blk_file_reading_column_Rho_title]
    Rho_box_0 = list(Rho_box_0)
    Rho_box_0 = np.transpose(Rho_box_0)
    Rho_box_0_mean = np.nanmean(Rho_box_0)

    volume_box_0 = data_box_0.loc[:, blk_file_reading_column_box_volume_title]
    volume_box_0 = list(volume_box_0)
    volume_box_0 = np.transpose(volume_box_0)
    volume_box_0_mean = np.nanmean(volume_box_0)
    length_if_cube_box_0_mean = (volume_box_0_mean) ** (1 / 3)

# add calculation for mole fractions
#    MF1_box_0 = data_box_0.loc[:,blk_file_reading_column_box_mf1_title]
#    MF1_box_0 = list(MF1_box_0)
#    MF1_box_0 = np.transpose(MF1_box_0)
#    MF1_box_0_mean = np.nanmean(MF1_box_0)
    
#    MF2_box_0 = data_box_0.loc[:,blk_file_reading_column_box_mf2_title]
#    MF2_box_0 = list(MF2_box_0)
#    MF2_box_0 = np.transpose(MF2_box_0)
#    MF2_box_0_mean = np.nanmean(MF2_box_0)


    # *************************
    # drawing in data from single file and extracting specific rows for the liquid box (end)
    # *************************


    # sort boxes based on density to liquid or vapor
#    pressure_box_liq_mean = pressure_box_0_mean
    total_molecules_box_liq_mean = total_molecules_box_0_mean
    Rho_box_liq_mean = Rho_box_0_mean
    volume_box_liq_mean = volume_box_0_mean
    length_if_cube_box_liq_mean = length_if_cube_box_0_mean
#    Hv_box_liq_mean = Hv_box_0_mean
#    Z_box_liq_mean = Z_box_0_mean
#    MF1_liq_mean = MF1_box_0_mean
#    MF2_liq_mean = MF2_box_0_mean
#    MF3_liq_mean = MF3_box_0_mean

    # *************************
    # drawing in data from single file and extracting specific rows for the vapor box (end)
    # *************************

    box_liq_replicate_data_txt_file = open(output_replicate_txt_file_name_liq, "w")
    box_liq_replicate_data_txt_file.write(
        f"{output_column_temp_title: <30} "
    #    f"{output_column_no_pressure_title: <30} "
        f"{output_column_total_molecules_title: <30} "
        f"{output_column_Rho_title: <30} "
        f"{output_column_box_volume_title: <30} "
        f"{output_column_box_length_if_cubed_title: <30} "
    #    f"{output_column_box_Hv_title: <30} "
#        f"{output_column_box_MF1_title: <30} "
#        f"{output_column_box_MF2_title: <30} "
#        f"{output_column_box_MF3_title: <30} "

        f" \n"
    )
    box_liq_replicate_data_txt_file.write(
        f"{job.sp.production_temperature_K: <30} "
    #    f"{pressure_box_liq_mean: <30} "
        f"{total_molecules_box_liq_mean: <30} "
        f"{Rho_box_liq_mean: <30} "
        f"{volume_box_liq_mean: <30} "
        f"{length_if_cube_box_liq_mean: <30} "
    #    f"{Hv_box_liq_mean: <30} "
#        f"{MF1_liq_mean: <30} "
#        f"{MF2_liq_mean: <30} "
#        f"{MF3_liq_mean: <30} "
        f" \n"
    )

    box_liq_replicate_data_txt_file.close()


    # ***********************
    # calc the avg data from the liq and vap boxes (end)
    # ***********************

# ******************************************************
# ******************************************************
# data analysis - get the average data from each replicate (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# data analysis - get the average and std. dev. from/across all the replicate (start)
# ******************************************************
# ******************************************************

@Project.pre(lambda *jobs: all(part_5a_analysis_individual_simulation_averages_completed(j)
                               for j in jobs[0]._project))
@Project.post(part_5b_analysis_replica_averages_completed)
@Project.operation(directives=
     {
         "np": 1,
         "ngpu": 0,
         "memory": memory_needed,
         "walltime": walltime_gomc_analysis_hr,
     }, aggregator=aggregator.groupby(key=statepoint_without_replica, sort_by="production_temperature_K", sort_ascending=False)
)
def part_5b_analysis_replica_averages(*jobs):
    # ***************************************************
    #  create the required lists and file labels total averages across the replicates (start)
    # ***************************************************
    # get the list used in this function
    temp_repilcate_list = []

   # pressure_repilcate_box_liq_list = []
    total_molecules_repilcate_box_liq_list = []
    Rho_repilcate_box_liq_list = []
    volume_repilcate_box_liq_list = []
    length_if_cube_repilcate_box_liq_list = []
    #Hv_repilcate_box_liq_list = []
    #Z_repilcate_box_liq_list = []
    MF1_replicate_box_liq_list = []
    MF2_replicate_box_liq_list = []

    output_column_temp_title = 'temp_K'  # column title title for temp
    #output_column_no_pressure_title = 'P_bar'  # column title title for PRESSURE
    output_column_total_molecules_title = "No_mol"  # column title title for TOT_MOL
    output_column_Rho_title = 'Rho_kg_per_m_cubed'  # column title title for TOT_DENS
    output_column_box_volume_title = 'V_ang_cubed'  # column title title for VOLUME
    output_column_box_length_if_cubed_title = 'L_m_if_cubed'  # column title title for VOLUME
    #output_column_box_Hv_title = 'Hv_kJ_per_mol'  # column title title for HEAT_VAP
    #output_column_box_Z_title = 'Z'  # column title title for compressiblity (Z)
#    output_column_box_MF1_title = 'Mol_Frac_F1A'  
#    output_column_box_MF2_title = 'Mol_Frac_C1A'  
#    output_column_box_MF3_title = 'Mol_Frac_FP'  

    output_column_temp_std_title = 'temp_std_K'  # column title title for temp
    #output_column_no_pressure_std_title = 'P_std_bar'  # column title title for PRESSURE
    output_column_total_molecules_std_title = "No_mol_std"  # column title title for TOT_MOL
    output_column_Rho_std_title = 'Rho_std_kg_per_m_cubed'  # column title title for TOT_DENS
    output_column_box_volume_std_title = 'V_std_ang_cubed'  # column title title for VOLUME
    output_column_box_length_if_cubed_std_title = 'L_std_m_if_cubed'  # column title title for VOLUME
    #output_column_box_Hv_std_title = 'Hv_std_kJ_per_mol'  # column title title for HEAT_VAP
    #output_column_box_Z_std_title = 'Z_std'  # column title title for compressiblity (Z)
#    output_column_box_MF1_std_title = 'MF1_std'  
#    output_column_box_MF2_std_title = 'MF2_std'  
#    output_column_box_MF3_std_title = 'MF3_std'  


    output_txt_file_header = f"{output_column_temp_title: <30} "\
                             f"{output_column_temp_std_title: <30} "\
                             f"{output_column_total_molecules_title: <30} "\
                             f"{output_column_total_molecules_std_title: <30} "\
                             f"{output_column_Rho_title: <30} "\
                             f"{output_column_Rho_std_title: <30} "\
                             f"{output_column_box_volume_title: <30} "\
                             f"{output_column_box_volume_std_title: <30} "\
                             f"{output_column_box_length_if_cubed_title: <30} "\
                             f"{output_column_box_length_if_cubed_std_title: <30} "\
                             f"\n"


    write_file_path_and_name_liq = f'analysis/{output_avg_std_of_replicates_txt_file_name_liq}'
    if os.path.isfile(write_file_path_and_name_liq):
        box_liq_data_txt_file = open(write_file_path_and_name_liq, "a")
    else:
        box_liq_data_txt_file = open(write_file_path_and_name_liq, "w")
        box_liq_data_txt_file.write(output_txt_file_header)

    # ***************************************************
    # create the required lists and file labels total averages across the replicates (end)
    # ***************************************************

    for job in jobs:

        # *************************
        # drawing in data from single simulation file and extracting specific
        # *************************
        reading_file_box_liq = job.fn(output_replicate_txt_file_name_liq)

        data_box_liq = pd.read_csv(reading_file_box_liq, sep='\s+', header=0, na_values='NaN', index_col=False)
        data_box_liq = pd.DataFrame(data_box_liq)

        #pressure_repilcate_box_liq = data_box_liq.loc[:, output_column_no_pressure_title][0]
        total_molecules_repilcate_box_liq = data_box_liq.loc[:, output_column_total_molecules_title][0]
        Rho_repilcate_box_liq = data_box_liq.loc[:, output_column_Rho_title][0]
        volume_repilcate_box_liq = data_box_liq.loc[:, output_column_box_volume_title][0]
        length_if_cube_repilcate_box_liq = (volume_repilcate_box_liq) ** (1 / 3)
        #Hv_repilcate_box_liq = data_box_liq.loc[:, output_column_box_Hv_title][0]
        #Z_repilcate_box_liq = data_box_liq.loc[:, output_column_box_Z_title][0]
#        MF1_replicate_box_liq = data_box_liq.loc[:, output_column_box_MF1_title][0]
#        MF2_replicate_box_liq = data_box_liq.loc[:, output_column_box_MF2_title][0]
#        MF3_replicate_box_liq = data_box_liq.loc[:, output_column_box_MF3_title][0]


        # *************************
        # drawing in data from bsingle file and extracting specific rows for the liquid box (end)
        # *************************

       
       
        # *************************
        # drawing in data from single file and extracting specific rows for the vapor box (end)
        # *************************
        temp_repilcate_list.append(job.sp.production_temperature_K)

        total_molecules_repilcate_box_liq_list.append(total_molecules_repilcate_box_liq)
        Rho_repilcate_box_liq_list.append(Rho_repilcate_box_liq)
        volume_repilcate_box_liq_list.append(volume_repilcate_box_liq)
        length_if_cube_repilcate_box_liq_list.append(length_if_cube_repilcate_box_liq)

    # *************************
    # get the replica means and std.devs (start)
    # *************************
    temp_mean = np.mean(temp_repilcate_list)
    temp_std = np.std(temp_repilcate_list, ddof=1)

    #pressure_mean_box_liq = np.mean(pressure_repilcate_box_liq_list)
    total_molecules_mean_box_liq = np.mean(total_molecules_repilcate_box_liq_list)
    Rho_mean_box_liq = np.mean(Rho_repilcate_box_liq_list)
    volume_mean_box_liq = np.mean(volume_repilcate_box_liq_list)
    length_if_cube_mean_box_liq = np.mean(length_if_cube_repilcate_box_liq_list)
    #Hv_mean_box_liq = np.mean(Hv_repilcate_box_liq_list)
    #Z_mean_box_liq = np.mean(Z_repilcate_box_liq_list)
    MF1_mean_box_liq = np.mean(MF1_replicate_box_liq_list)
    MF2_mean_box_liq = np.mean(MF2_replicate_box_liq_list)

    #pressure_std_box_liq = np.std(pressure_repilcate_box_liq_list, ddof=1)
    total_molecules_std_box_liq = np.std(total_molecules_repilcate_box_liq_list, ddof=1)
    Rho_std_box_liq= np.std(Rho_repilcate_box_liq_list, ddof=1)
    volume_std_box_liq = np.std(volume_repilcate_box_liq_list, ddof=1)
    length_if_cube_std_box_liq = np.std(length_if_cube_repilcate_box_liq_list, ddof=1)
    #Hv_std_box_liq = np.std(Hv_repilcate_box_liq_list, ddof=1)
    #Z_std_box_liq = np.std(Z_repilcate_box_liq_list, ddof=1)
    MF1_std_box_liq = np.std(MF1_replicate_box_liq_list,ddof=1)
    MF2_std_box_liq = np.std(MF2_replicate_box_liq_list,ddof=1)

    # *************************
    # get the replica means and std.devs (end)
    # *************************


    # ************************************
    # write the analysis data files for the liquid and vapor boxes (start)
    # ************************************

    box_liq_data_txt_file.write(
        f"{temp_mean: <30} "
        f"{temp_std: <30} "
    #    f"{pressure_mean_box_liq: <30} "
    #    f"{pressure_std_box_liq: <30} "
        f"{total_molecules_mean_box_liq: <30} "
        f"{total_molecules_std_box_liq: <30} "
        f"{Rho_mean_box_liq: <30} "
        f"{Rho_std_box_liq: <30} "
        f"{volume_mean_box_liq: <30} "
        f"{volume_std_box_liq: <30} "
        f"{length_if_cube_mean_box_liq: <30} "
        f"{length_if_cube_std_box_liq: <30} "
    #    f"{Hv_mean_box_liq: <30} "
    #    f"{Hv_std_box_liq: <30} "
        f"{MF1_mean_box_liq: <30} "
        f"{MF1_std_box_liq: <30} "
        f"{MF2_mean_box_liq: <30} "
        f"{MF2_std_box_liq: <30} "
        f" \n"
    )

   
    # ************************************
    # write the analysis data files for the liquid and vapor boxes (end)
    # ************************************

# ******************************************************
# ******************************************************
# signac end code (start)
# ******************************************************
# ******************************************************
if __name__ == "__main__":
    pr = Project()
    pr.main()
# *****************************************************
