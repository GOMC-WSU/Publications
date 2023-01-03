"""GOMC's setup for signac, signac-flow, signac-dashboard for this study."""
# project.py


import flow
# from flow.environment import StandardEnvironment
import mbuild as mb
import mosdef_gomc.formats.gmso_charmm_writer as mf_charmm
import mosdef_gomc.formats.gmso_gomc_conf_writer as gomc_control
import numpy as np

from alchemlyb.parsing.gomc import  extract_dHdl,  extract_u_nk
from alchemlyb.estimators import MBAR, BAR, TI
import alchemlyb.preprocessing.subsampling as ss
import pandas as pd
import numpy as np
import os

import unyt as u
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment

from src.utils.forcefields import get_ff_path
from src.utils.forcefields import get_molecule_path
from templates.NAMD_conf_template import generate_namd_equilb_control_file


class Project(FlowProject):
    """Subclass of FlowProject to provide custom methods and attributes."""

    def __init__(self):
        super().__init__()


class Grid(DefaultSlurmEnvironment):  # Grid(StandardEnvironment):
    """Subclass of DefaultSlurmEnvironment for WSU's Grid cluster."""

    hostname_pattern = r".*\.grid\.wayne\.edu"
    template = "grid.sh"



# ******************************************************
# users typical variables, but not all (start)
# ******************************************************
# set binary path to gomc binary files (the bin folder).
# If the gomc binary files are callable directly from the terminal without a path,
# please just enter and empty string (i.e., "" or '')

# Enter the NAMD and GOMC binary path here (MANDATORY INFORMAION)
gomc_binary_path = "/home/brad/Programs/GOMC/GOMC_2_76/bin"
namd_binary_path = "/home/brad/Programs/NAMD/NAMD_2.14_RTX_3080_build_Source_CUDA"

# number of simulation steps
gomc_steps_equilb_design_ensemble = 10 * 10**6 # set value for paper = 10 * 10**6
gomc_steps_lamda_production = 50 * 10**6 # set value for paper = 50 * 10**6

gomc_output_data_every_X_steps = 100 * 10**3 # set value for paper = 100 * 10**3
gomc_free_energy_output_data_every_X_steps = 10 * 10**3 # set value for paper = 10 * 10**3


# Free energy calcs: set free energy data in doc
# this number will generate the lamdas
# set the number of lambda spacings, which includes 0 to 1
number_of_lambda_spacing_including_zero_int = 11


# force field (FF) file for all simulations in that job
# Note: do not add extensions
namd_ff_filename_str = "in_namd_FF"
gomc_ff_filename_str = "in_gomc_FF"

# initial mosdef structure and coordinates
# Note: do not add extensions
mosdef_structure_box_0_name_str = "mosdef_box_0"

# melt equilb simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
namd_equilb_NPT_control_file_name_str = "namd_equilb_NPT"

# The equilb using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
gomc_equilb_design_ensemble_control_file_name_str = "gomc_equilb_design_ensemble"

# The production run using the ensemble used for the simulation design, which
# includes the simulation runs GOMC control file input and simulation outputs
# Note: do not add extensions
gomc_production_control_file_name_str = "gomc_production_run"

# Analysis (each replicates averages):
# Output text (txt) file names for each replicates averages
# directly put in each replicate folder (.txt, .dat, etc)
output_replicate_txt_file_name_box_0 = "analysis_avg_data_box_0.txt"

# Analysis (averages and std. devs. of  # all the replcates):
# Output text (txt) file names for the averages and std. devs. of all the replcates,
# including the extention (.txt, .dat, etc)
output_avg_std_of_replicates_txt_file_name_box_0 = "analysis_avg_std_of_replicates_box_0.txt"



walltime_mosdef_hr = 24
walltime_namd_hr = 24
walltime_gomc_equilbrium_hr = 72
walltime_gomc_production_hr = 368
walltime_gomc_analysis_hr = 4
memory_needed = 16



# forcefield names dict
forcefield_residue_to_ff_filename_dict = {
    "TIP4": "tip4p_2005.xml",
    "Ne": "nobel_gas_vrabec_LB_mixing.xml",
    "Rn": "nobel_gas_vrabec_LB_mixing.xml",
}


# smiles of mol2 file input a .mol2 file or smiles as a string
smiles_or_mol2_name_to_value_dict = {
    "TIP4": 'tip4p.mol2',
    "Ne": "Ne",
    "Rn": "Rn",
}


# get the paths to the smiles or mol2 files
smiles_or_mol2 = {}
for smiles_or_mol2_iter_i in list(smiles_or_mol2_name_to_value_dict.keys()):
    smiles_or_mol2.update(
        {str(smiles_or_mol2_iter_i):
             {"use_smiles": get_molecule_path(
                 str(smiles_or_mol2_name_to_value_dict[str(smiles_or_mol2_iter_i)]))[0],
              "smiles_or_mol2": get_molecule_path(
                  str(smiles_or_mol2_name_to_value_dict[str(smiles_or_mol2_iter_i)]))[1],
              }
         }
    )

# get the paths to the FF xmls
forcefield_dict = {}
for forcefield_dict_iter_i in list(forcefield_residue_to_ff_filename_dict.keys()):
    forcefield_dict.update(
        {str(forcefield_dict_iter_i): get_ff_path(
            forcefield_residue_to_ff_filename_dict[str(forcefield_dict_iter_i)])
        }
    )
print("*********************")
print("*********************")
print("smiles_or_mol2 = " +str(smiles_or_mol2))
print("forcefield_dict = " +str(forcefield_dict))
print("*********************")
print("*********************")

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
# functions for free energy calcs MBAR, TI, and BAR for getting delta free energy and delta error (start)
# ******************************************************
# ******************************************************

def get_delta_TI_or_MBAR(TI_or_MBAR_estimate, k_b_T):
    """ Return the change in free energy and standard deviation for the MBAR and TI estimates.

    """
    delta = TI_or_MBAR_estimate.delta_f_.iloc[0, -1] * k_b_T
    std_delta = TI_or_MBAR_estimate.d_delta_f_.iloc[0, -1] * k_b_T
    return delta, std_delta


def get_delta_BAR(BAR_estimate, k_b_T):
    """ Return the change in free energy and standard deviation for the BAR estimates.

    """
    error_estimate = 0.0

    for i in range(len(BAR_estimate.d_delta_f_) - 1):
        error_estimate += BAR_estimate.d_delta_f_.values[i][i + 1] ** 2

    delta = BAR_estimate.delta_f_.iloc[0, -1] * k_b_T
    std_delta = k_b_T * error_estimate ** 0.5
    return delta, std_delta

# ******************************************************
# ******************************************************
# functions for free energy calcs MBAR, TI, and BAR for getting delta free energy and delta error (end)
# ******************************************************
# ******************************************************

@Project.label
def part_1a_initial_data_input_to_json(job):
    """Check that the initial job data is written to the json files."""
    data_written_bool = False
    if job.isfile(f"{'signac_job_document.json'}"):
        data_written_bool = True

    return data_written_bool


@Project.post(part_1a_initial_data_input_to_json)
@Project.operation.with_directives(
    {
        "np": 1,
        "ngpu": 0,
        "memory": memory_needed,
        "walltime": walltime_mosdef_hr,
    }
)
@flow.with_job
def initial_parameters(job):
    """Set the initial job parameters into the jobs doc json file."""
    # select

    # set free energy data in doc
    # Free energy calcs
    # lamda generator

    LambdaVDW_list = []
    InitialState_list = []
    for lamda_i in range(0, int(number_of_lambda_spacing_including_zero_int)):
        lambda_space_increments = 1 / int(number_of_lambda_spacing_including_zero_int - 1)
        LambdaVDW_list.append(np.round(lamda_i * lambda_space_increments, decimals=8))
        InitialState_list.append(lamda_i)
    print("*********************")
    print("*********************")
    print("LambdaVDW_list = " + str(LambdaVDW_list))
    print("InitialState_list = " + str(InitialState_list))
    print("*********************")
    print("*********************")
    if LambdaVDW_list[0] != 0 and LambdaVDW_list[-1] != 1 :
        raise ValueError("ERROR: The selected lambda list values do not start with a 0 and end 1.")

    job.doc.LambdaVDW_list = LambdaVDW_list
    job.doc.InitialState_list = InitialState_list

    # set the GOMC production ensemble temp, pressure, molecule, box dimenstion and residue names
    job.doc.production_ensemble = "NVT"
    job.doc.production_pressure_bar = (1 * u.atm).to('bar')
    job.doc.production_temperature_K = job.sp.production_temperature_K

    job.doc.N_liquid_solvent = 1000
    job.doc.N_liquid_solute = 1

    job.doc.liq_box_lengths_ang = 31.07 * u.angstrom

    job.doc.Rcut_ang = 15 * u.angstrom  # this is the Rcut for GOMC it is the Rswitch for NAMD
    job.doc.Rcut_for_switch_namd_ang = 17 * u.angstrom  # Switch Rcut for NAMD's Switch function
    job.doc.neighbor_list_dist_namd_ang = 22 * u.angstrom # NAMD's neighbor list

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

    # set solvent and solute in doc
    job.doc.solvent = "TIP4"
    job.doc.solute = job.sp.solute

    # set rcut, ewalds
    if job.doc.solvent in ["TIP4", "TIP3"] and job.doc.solute in ["He", "Ne", "Kr", "Ar", "Xe", "Rn"]:
        job.doc.namd_node_ncpu = 1
        job.doc.namd_node_ngpu = 1

        job.doc.gomc_ncpu = 1  # 1 is optimal but I want data quick.  run time is set for 1 cpu
        job.doc.gomc_ngpu = 0

    else:
        raise ValueError(
            "ERROR: The solvent and solute do are not set up to selected the mixing rules or electrostatics "
        )

    # get the namd binary paths
    if job.doc.namd_node_ngpu == 0:
        job.doc.namd_cpu_or_gpu = "CPU"

    elif job.doc.namd_node_ngpu == 1:
        job.doc.namd_cpu_or_gpu = "GPU"

    else:
        raise ValueError(
            "Tee NAMD CPU and GPU can not be determined as force field (FF) is not available in the selection, "
            "or GPU selection is is not 0 or 1."
        )

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

    # set the initial iteration number of the simulation
    job.doc.gomc_equilb_design_ensemble_dict = {}
    job.doc.gomc_production_run_ensemble_dict = {}


    if job.doc.production_ensemble == "NPT":
        job.doc.namd_equilb_NPT_gomc_binary_file = f"namd2"
        job.doc.gomc_equilb_design_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NPT"
        job.doc.gomc_production_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NPT"

    elif job.doc.production_ensemble == "NVT":
        job.doc.namd_equilb_NPT_gomc_binary_file = f"namd2"
        job.doc.gomc_equilb_design_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NPT"
        job.doc.gomc_production_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_NVT"

    else:
        raise ValueError(
            "ERROR: The 'GCMC', 'GEMC_NVT', 'GEMC_NPT' ensembles is not currently available for this project.py "
        )


# ******************************************************
# ******************************************************
# create some initial variable to be store in each jobs
# directory in an additional json file, and test
# to see if they are written (end).
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC psf, pdb, and force field (FF) files were written (start)
# ******************************************************
# ******************************************************

# check if GOMC-MOSDEF wrote the gomc files
# @Project.pre(select_production_ensemble)
@Project.label
@flow.with_job
def mosdef_input_written(job):
    """Check that the mosdef files (psf, pdb, and force field (FF) files) are written ."""
    file_written_bool = False

    if (
        job.isfile(f"{namd_ff_filename_str}.inp")
        and job.isfile(f"{gomc_ff_filename_str}.inp")
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
        with open(job.fn(f"{control_file}"), "r") as fp:
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "OutputName" in line:
                    split_move_line = line.split()
                    if split_move_line[0] == "OutputName":
                        file_written_bool = True

    return file_written_bool

# function for checking if the NAMD control file is written
def namd_control_file_written(job, control_filename_str):
    """General check that the NAMD control files are written."""
    file_written_bool = False
    control_file = f"{control_filename_str}.conf"
    if job.isfile(control_file):
        with open(job.fn(f"{control_file}"), "r") as fp:
            out_namd = fp.readlines()
            for i, line in enumerate(out_namd):
                if "cellBasisVector1" in line:
                    split_move_line = line.split()
                    if split_move_line[0] == "cellBasisVector1":
                        file_written_bool = True

    return file_written_bool


# checking if the NAMD control file is written for the melt equilb NVT run
@Project.label
@flow.with_job
def part_2a_namd_equilb_NPT_control_file_written(job):
    """General check that the namd_equilb_NPT_control_file
    (high temperature to set temp NAMD control file) is written."""
    return namd_control_file_written(job, namd_equilb_NPT_control_file_name_str)

# checking if the GOMC control file is written for the equilb run with the selected ensemble
@Project.label
@flow.with_job
def part_2b_gomc_equilb_design_ensemble_control_file_written(job):
    """General check that the gomc_equilb_design_ensemble (run temperature) gomc control file is written."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                gomc_control_file_written(
                    job,
                    job.doc.gomc_equilb_design_ensemble_dict[
                        str(initial_state_i)
                    ]["output_name_control_file_name"],
                )
            except:
                return False
        return True
    except:
        return False

# checking if the GOMC control file is written for the production run
@Project.label
@flow.with_job
def part_2c_gomc_production_control_file_written(job):
    """General check that the gomc_production_control_file (run temperature) is written."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                return gomc_control_file_written(
                    job,
                    job.doc.gomc_production_run_ensemble_dict[
                        str(initial_state_i)
                    ]["output_name_control_file_name"],
                )
            except:
                return False
        return True
    except:
        return False

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
    if job.isfile("out_{}.dat".format(control_filename_str)) and job.isfile(
        "{}_merged.psf".format(control_filename_str)
    ):
        output_started_bool = True

    return output_started_bool

# function for checking if NAMD simulations are started
def namd_simulation_started(job, control_filename_str):
    """General check to see if the namd simulation is started."""
    output_started_bool = False
    if job.isfile("out_{}.dat".format(control_filename_str)) and job.isfile(
        "{}.restart.xsc".format(control_filename_str)
    ):
        output_started_bool = True

    return output_started_bool


# check if melt equilb_NVT namd run is started
@Project.label
@flow.with_job
def part_3a_output_namd_equilb_NPT_started(job):
    """Check to see if the namd_equilb_NPT_control_file is started
    (high temperature to set temperature in NAMD control file)."""
    return namd_simulation_started(job, namd_equilb_NPT_control_file_name_str)


# check if equilb_with design ensemble GOMC run is started
@Project.label
@flow.with_job
def part_3b_output_gomc_equilb_design_ensemble_started(job):
    """Check to see if the gomc_equilb_design_ensemble simulation is started (set temperature)."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                if job.isfile(
                    "out_{}.dat".format(
                        job.doc.gomc_equilb_design_ensemble_dict[
                            str(initial_state_i)
                        ]["output_name_control_file_name"]
                    )
                ):
                    gomc_simulation_started(
                        job,
                        job.doc.gomc_equilb_design_ensemble_dict[
                            str(initial_state_i)
                        ]["output_name_control_file_name"],
                    )

                else:
                    return False
            except:
                return False

        return True
    except:
        return False

# check if production GOMC run is started by seeing if the GOMC consol file and the merged psf exist
@Project.label
@flow.with_job
def part_part_3c_output_gomc_production_run_started(job):
    """Check to see if the gomc production run simulation is started (set temperature)."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                if job.isfile(
                    "out_{}.dat".format(
                        job.doc.gomc_production_run_ensemble_dict[
                            str(initial_state_i)
                        ]["output_name_control_file_name"]
                    )
                ):
                    gomc_simulation_started(
                        job,
                        job.doc.gomc_production_run_ensemble_dict[
                            str(initial_state_i)
                        ]["output_name_control_file_name"],
                    )
                else:
                    return False
            except:
                return False
        return True
    except:
        return False
# ******************************************************
# ******************************************************
# check if GOMC simulations started (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC and NAMD simulation are completed properly (start)
# ******************************************************
# ******************************************************
# function for checking if GOMC simulations are completed properly
def gomc_sim_completed_properly(job, control_filename_str):
    """General check to see if the gomc simulation was completed properly."""
    job_run_properly_bool = False
    output_log_file = "out_{}.dat".format(control_filename_str)
    if job.isfile(output_log_file):
        with open(job.fn(f"{output_log_file}"), "r") as fp:
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "Move" in line:
                    split_move_line = line.split()
                    if (
                        split_move_line[0] == "Move"
                        and split_move_line[1] == "Type"
                        and split_move_line[2] == "Mol."
                        and split_move_line[3] == "Kind"
                    ):
                        job_run_properly_bool = True
    else:
        job_run_properly_bool = False

    return job_run_properly_bool

# function for checking if NAMD simulations are completed properly
def namd_sim_completed_properly(job, control_filename_str):
    """General check to see if the namd simulation was completed properly."""
    job_run_properly_bool = False
    output_log_file = "out_{}.dat".format(control_filename_str)
    if job.isfile(output_log_file):
        with open(job.fn(f"{output_log_file}"), "r") as fp:
            out_namd = fp.readlines()
            for i, line in enumerate(out_namd):
                if "WallClock:" in line:
                    split_move_line = line.split()
                    if (split_move_line[0] == "WallClock:"
                            and split_move_line[2] == "CPUTime:"
                            and split_move_line[4] == "Memory:"
                    ):
                        job_run_properly_bool = True
    else:
        job_run_properly_bool = False

    return job_run_properly_bool

# check if melt equilb NVT GOMC run completed by checking the end of the GOMC consol file
@Project.label
@flow.with_job
def part_4a_job_namd_equilb_NPT_completed_properly(job):
    """Check to see if the  namd_equilb_NPT_control_file was completed properly
    (high temperature to set temperature NAMD control file)."""
    x = namd_sim_completed_properly(
        job, namd_equilb_NPT_control_file_name_str
    )
    #print(f'namd check = {x}')
    return namd_sim_completed_properly(
        job, namd_equilb_NPT_control_file_name_str
    )


# check if equilb selected ensemble GOMC run completed by checking the end of the GOMC consol file
@Project.label
@flow.with_job
def part_4b_job_gomc_equilb_design_ensemble_completed_properly(job):
    """Check to see if the gomc_equilb_design_ensemble simulation was completed properly (set temperature)."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                filename_4b_iter = job.doc.gomc_equilb_design_ensemble_dict[
                    str(initial_state_i)
                ]["output_name_control_file_name"]

                if gomc_sim_completed_properly(
                    job,
                    filename_4b_iter,
                ) is False:
                    return False
            except:
                return False
        return True
    except:
        return False

# check if production GOMC run completed by checking the end of the GOMC consol file
@Project.label
@flow.with_job
def part_4c_job_production_run_completed_properly(job):
    """Check to see if the gomc production run simulation was completed properly (set temperature)."""
    try:
        for initial_state_i in list(job.doc.InitialState_list):
            try:
                filename_4c_iter = job.doc.gomc_production_run_ensemble_dict[
                    str(initial_state_i)
                ]["output_name_control_file_name"]
                if gomc_sim_completed_properly(
                    job,
                    filename_4c_iter,
                ) is False:
                    return False

                # check specifically for the FE files
                if job.isfile(f'Free_Energy_BOX_0_{filename_4c_iter}.dat') is False:
                    return False

            except:
                return False
        return True
    except:
        return False

# ******************************************************
# ******************************************************
# check if GOMC and NAMD simulation are completed properly (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC anaylsis is completed properly (start)
# ******************************************************
# ******************************************************

# check if analysis is done for the individual replicates wrote the gomc files
@Project.pre(part_4c_job_production_run_completed_properly)
@Project.label
@flow.with_job
def part_5a_analysis_individual_simulation_averages_completed(job):
    """Check that the individual simulation averages files are written ."""
    file_written_bool = False
    if (
        job.isfile(
            f"{output_replicate_txt_file_name_box_0}"
        )
    ):
        file_written_bool = True

    return file_written_bool


# check if analysis for averages of all the replicates is completed
@Project.pre(part_5a_analysis_individual_simulation_averages_completed)
@Project.label
def part_5b_analysis_replica_averages_completed(*jobs):
    """Check that the simulation replicate average and std. dev. files are written."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_avg_std_of_replicates_txt_file_name_box_0}"
            )
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return all_file_written_bool_pass


# ******************************************************
# ******************************************************
# check if GOMC anaylsis is completed properly (end)
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
    mbuild_box_seed_no = job.doc.replica_number_int

    solvent = mb.load(smiles_or_mol2[job.doc.solvent]['smiles_or_mol2'],
                      smiles=smiles_or_mol2[job.doc.solvent]['use_smiles']
                      )
    solvent.name = job.doc.solvent

    if job.doc.solvent not in ["TIP4"]:
        solvent.energy_minimize(forcefield=forcefield_dict[job.doc.solvent], steps=10 ** 5)

    if job.sp.solute in ["He", "Ne", "Kr", "Ar", "Xe", "Rn"]:
        solute = mb.Compound(name=job.doc.solute)
    else:
        solute = mb.load(smiles_or_mol2[job.sp.solute]['smiles_or_mol2'],
                         smiles=smiles_or_mol2[job.sp.solute]['use_smiles']
                         )
    solute.name = job.sp.solute

    # only put the FF molecules in the simulation in the dictionaly input into the Chamm object.
    minimal_forcefield_dict = {solute.name: forcefield_dict[solute.name],
                               solvent.name: forcefield_dict[solvent.name]
                               }

    solute.energy_minimize(forcefield=forcefield_dict[job.sp.solute], steps=10 ** 5)

    bead_to_atom_name_dict = {
        "_LP": "LP",
    }
    residues_list = [solute.name, solvent.name]
    print("residues_list  = " +str(residues_list ))

    if job.doc.solvent in ["TIP4", "TIP3"]:
        gomc_fix_bonds_angles_residues_list = [solvent.name]
    else:
        gomc_fix_bonds_angles_residues_list  = None

    print('Running: filling liquid box')
    box_0 = mb.fill_box(compound=[solute, solvent],
                        n_compounds=[job.doc.N_liquid_solute, job.doc.N_liquid_solvent],
                        box=[u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("nm"),
                             u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("nm"),
                             u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("nm"),
                             ],
                        seed=mbuild_box_seed_no
                        )
    print('Completed: filling liquid box')

    print('Running: GOMC FF file, and the psf and pdb files')
    if job.doc.production_ensemble in ["NVT", "NPT"]:
        print('Running: namd_charmm')
        namd_charmm = mf_charmm.Charmm(
            box_0,
            mosdef_structure_box_0_name_str,
            structure_box_1=None,
            filename_box_1=None,
            ff_filename= namd_ff_filename_str,
            forcefield_selection=minimal_forcefield_dict,
            residues=residues_list,
            bead_to_atom_name_dict=bead_to_atom_name_dict,
            gomc_fix_bonds_angles=None,
        )

        print('Running: gomc_charmm')
        gomc_charmm = mf_charmm.Charmm(
            box_0,
            mosdef_structure_box_0_name_str,
            structure_box_1=None,
            filename_box_1=None,
            ff_filename=  gomc_ff_filename_str,
            forcefield_selection=minimal_forcefield_dict,
            residues=residues_list,
            bead_to_atom_name_dict=bead_to_atom_name_dict,
            gomc_fix_bonds_angles=gomc_fix_bonds_angles_residues_list,
        )

    else:
        raise ValueError("ERROR: The GCMC and GEMC ensembles are not supported in this script.")

    if write_files == True:
        gomc_charmm.write_inp()

        namd_charmm.write_inp()

        namd_charmm.write_psf()

        namd_charmm.write_pdb()

    print("#**********************")
    print("Completed: GOMC Charmm Object")
    print("#**********************")

    return [namd_charmm, gomc_charmm]


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
@Project.post(part_2a_namd_equilb_NPT_control_file_written)
@Project.post(part_2b_gomc_equilb_design_ensemble_control_file_written)
@Project.post(part_2c_gomc_production_control_file_written)
@Project.post(mosdef_input_written)
@Project.operation.with_directives(
    {
        "np": 1,
        "ngpu": 0,
        "memory": memory_needed,
        "walltime": walltime_mosdef_hr,
    }
)
@flow.with_job
def build_psf_pdb_ff_gomc_conf(job):
    """Build the Charmm object and write the pdb, psd, and force field (FF)
    files for all the simulations in the workspace."""
    [namd_charmm_object_with_files, gomc_charmm_object_with_files] = build_charmm(job, write_files=True)

    FreeEnergyCalc = [True, int(gomc_free_energy_output_data_every_X_steps)]
    MoleculeType = [job.sp.solute, 1]

    use_ElectroStatics = True
    VDWGeometricSigma = False
    Exclude = "1-4"

    # common variables
    cutoff_style = "VDW"
    if cutoff_style != "VDW":
        raise ValueError("ERROR: this project is only set up for the SWITCH cutoff style for NAMD"
                         "and VDW for GOMC.  Therefore, the cutoff style selected must be VDW. "
                         "Rswitch for namd only so the r_switch_dist_start and "
                         "r_switch_dist_end must be supplied for NAMD. GOMC will then move to VDW "
                         "with the switch dist (r_switch_dist_start) as the cutoff with LRC.")

    production_temperature_K = job.sp.production_temperature_K * u.K

    production_pressure_bar = job.doc.production_pressure_bar * u.bar

    box_lengths_ang = [u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("angstrom"),
                       u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("angstrom"),
                       u.unyt_quantity(job.doc.liq_box_lengths_ang, 'angstrom').to_value("angstrom"),
                       ]

    seed_no = job.doc.replica_number_int

    namd_template_path_str = os.path.join(project_directory_path, "templates/NAMD_conf_template.conf")

    if job.doc.solvent in ["TIP3"] or job.sp.solute in ["TIP3"]:
        namd_uses_water = True
        namd_water_model = 'tip3'
    elif job.doc.solvent in ["TIP4"] or job.sp.solute in ["TIP4"]:
        namd_uses_water = True
        namd_water_model = 'tip4'
    else:
        namd_uses_water = False
        namd_water_model= None

    # generate the namd file
    # NOTE: the production and melt temps are converted to intergers so they can be ramped down
    # from hot to cool to equilibrate the system.
    generate_namd_equilb_control_file(template_path_filename=namd_template_path_str,
                                      namd_path_conf_filename=namd_equilb_NPT_control_file_name_str,
                                      namd_path_file_output_names=namd_equilb_NPT_control_file_name_str,
                                      namd_uses_water=namd_uses_water,
                                      namd_water_model=namd_water_model,
                                      namd_electrostatics_bool=use_ElectroStatics,
                                      namd_vdw_geometric_sigma_bool=VDWGeometricSigma,
                                      namd_psf_path_filename=f"{mosdef_structure_box_0_name_str}.psf",
                                      namd_pdb_path_filename=f"{mosdef_structure_box_0_name_str}.pdb",
                                      namd_ff_path_filename=f"{namd_ff_filename_str}.inp",
                                      namd_production_temp_K= int(production_temperature_K.to_value("K")),
                                      namd_production_pressure_bar=production_pressure_bar.to_value("bar"),
                                      electrostatic_1_4=namd_charmm_object_with_files.electrostatic_1_4,
                                      non_bonded_cutoff=job.doc.Rcut_for_switch_namd_ang,
                                      non_bonded_switch_distance=job.doc.Rcut_ang,
                                      pairlist_distance=job.doc.neighbor_list_dist_namd_ang,
                                      box_lengths=box_lengths_ang,
                                      )

    print("#**********************")
    print("Completed: namd_equilb_NPT GOMC control file writing")
    print("#**********************")
    # ******************************************************
    # namd_equilb_NPT - psf, pdb, force field (FF) file writing and GOMC control file writing  (end)
    # ******************************************************


    # ******************************************************
    # equilb selected_ensemble, if NVT -> NPT - GOMC control file writing  (start)
    # Note: the control files are written for the max number of gomc_equilb_design_ensemble runs
    # so the Charmm object only needs created 1 time.
    # ******************************************************
    print("#**********************")
    print("Started: equilb NPT or GEMC-NVT GOMC control file writing")
    print("#**********************")

    for initial_state_sims_i in list(job.doc.InitialState_list):
        namd_restart_pdb_psf_file_name_str = mosdef_structure_box_0_name_str

        restart_control_file_name_str = namd_equilb_NPT_control_file_name_str
        output_name_control_file_name = "{}_initial_state_{}".format(
            gomc_equilb_design_ensemble_control_file_name_str, initial_state_sims_i
        )

        job.doc.gomc_equilb_design_ensemble_dict.update(
            {
                initial_state_sims_i: {
                    "restart_control_file_name": restart_control_file_name_str,
                    "output_name_control_file_name": output_name_control_file_name,
                }
            }
        )

        # calc MC steps
        MC_steps = int(gomc_steps_equilb_design_ensemble)
        EqSteps = 1000

        # output all data and calc frequecy
        output_true_list_input = [
            True,
            int(gomc_output_data_every_X_steps),
        ]
        output_false_list_input = [
            False,
            int(gomc_output_data_every_X_steps),
        ]

        if job.doc.solvent in ["TIP4", "TIP3"] \
                and job.doc.solute in ["He", "Ne", "Kr", "Ar", "Xe", "Rn"]:
            used_ensemble = "NPT"
            if job.doc.production_ensemble in ["NVT", "NPT"]:
                VolFreq = (0.01,)
                MultiParticleFreq = (None,)
                IntraSwapFreq = (0.0,)
                CrankShaftFreq = (None,)
                SwapFreq = (None,)
                DisFreq = (0.39,)
                RotFreq = (0.3,)
                RegrowthFreq = (0.3,)

            else:
                raise ValueError(
                    "Moleules MC move ratios not listed for this solvent and solute or ensemble "
                    "in the GOMC control file writer."
                )

            Coordinates_box_0 = "{}.pdb".format(
                namd_restart_pdb_psf_file_name_str
            )
            Structure_box_0 = "{}.psf".format(
                namd_restart_pdb_psf_file_name_str
            )
            binCoordinates_box_0 = "{}.restart.coor".format(
                restart_control_file_name_str
            )
            extendedSystem_box_0 = "{}.restart.xsc".format(
                restart_control_file_name_str
            )

        gomc_control.write_gomc_control_file(
            gomc_charmm_object_with_files,
            output_name_control_file_name,
            used_ensemble,
            MC_steps,
            production_temperature_K,
            ff_psf_pdb_file_directory=None,
            check_input_files_exist=False,
            Parameters="{}.inp".format(gomc_ff_filename_str),
            Restart=True,
            Checkpoint=False,
            ExpertMode=False,
            Coordinates_box_0=Coordinates_box_0,
            Structure_box_0=Structure_box_0,
            binCoordinates_box_0=binCoordinates_box_0,
            extendedSystem_box_0=extendedSystem_box_0,
            binVelocities_box_0=None,
            Coordinates_box_1=None,
            Structure_box_1=None,
            binCoordinates_box_1=None,
            extendedSystem_box_1=None,
            binVelocities_box_1=None,
            input_variables_dict={
                "PRNG": seed_no,
                "Pressure": production_pressure_bar,
                "Ewald": use_ElectroStatics,
                "ElectroStatic": use_ElectroStatics,
                "VDWGeometricSigma": VDWGeometricSigma,
                "Rcut": (job.doc.Rcut_ang * u.angstrom).to("angstrom"),
                "Exclude": Exclude,
                "VolFreq": VolFreq[-1],
                "MultiParticleFreq": MultiParticleFreq[-1],
                "IntraSwapFreq": IntraSwapFreq[-1],
                "CrankShaftFreq": CrankShaftFreq[-1],
                "SwapFreq": SwapFreq[-1],
                "DisFreq": DisFreq[-1],
                "RotFreq": RotFreq[-1],
                "RegrowthFreq": RegrowthFreq[-1],
                "OutputName": output_name_control_file_name,
                "EqSteps": EqSteps,
                "PressureCalc": output_false_list_input,
                "RestartFreq": output_true_list_input,
                "CheckpointFreq": output_true_list_input,
                "ConsoleFreq": output_true_list_input,
                "BlockAverageFreq": output_true_list_input,
                "HistogramFreq": output_false_list_input,
                "CoordinatesFreq": output_false_list_input,
                "DCDFreq": output_true_list_input,
                "Potential": cutoff_style,
                "LRC": True,
                "RcutLow": 0 * u.angstrom,
                "CBMC_First": 12,
                "CBMC_Nth": 10,
                "CBMC_Ang": 50,
                "CBMC_Dih": 50,
                "FreeEnergyCalc": FreeEnergyCalc,
                "MoleculeType": MoleculeType,
                "InitialState": initial_state_sims_i,
                "LambdaVDW": list(job.doc.LambdaVDW_list),
                # "LambdaCoulomb": None,
            },
        )
        print("#**********************")
        print("Completed: equilb NPT or GEMC-NVT GOMC control file writing")
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
        print("Started: production NPT or GEMC-NVT GOMC control file writing")
        print("#**********************")

        output_name_control_file_name = "{}_initial_state_{}".format(
            gomc_production_control_file_name_str, initial_state_sims_i
        )
        restart_control_file_name_str = "{}_initial_state_{}".format(
            gomc_equilb_design_ensemble_control_file_name_str, int(initial_state_sims_i)
        )
        job.doc.gomc_production_run_ensemble_dict.update(
            {
                initial_state_sims_i: {
                    "restart_control_file_name": restart_control_file_name_str,
                    "output_name_control_file_name": output_name_control_file_name,
                }
            }
        )

        # calc MC steps
        MC_steps = int(gomc_steps_lamda_production)
        EqSteps = 1000


        # output all data and calc frequecy
        output_true_list_input = [
            True,
            int(gomc_output_data_every_X_steps),
        ]
        output_false_list_input = [
            False,
            int(gomc_output_data_every_X_steps),
        ]
        
        
        if job.doc.solvent in ["TIP4", "TIP3"] \
                    and job.doc.solute in ["He", "Ne", "Kr", "Ar", "Xe", "Rn"]:
            used_ensemble = job.doc.production_ensemble
            if job.doc.production_ensemble in ["NVT", "NPT"]:
                if job.doc.production_ensemble in ["NVT"]:
                    VolFreq = (0.00,)
                    MultiParticleFreq = (None,)
                    IntraSwapFreq = (0.0,)
                    CrankShaftFreq = (None,)
                    SwapFreq = (None,)
                    DisFreq = (0.4,)
                    RotFreq = (0.3,)
                    RegrowthFreq = (0.3,)

                elif job.doc.production_ensemble in ["NPT"]:
                    VolFreq = (0.01,)
                    MultiParticleFreq = (None,)
                    IntraSwapFreq = (0.0,)
                    CrankShaftFreq = (None,)
                    SwapFreq = (None,)
                    DisFreq = (0.39,)
                    RotFreq = (0.3,)
                    RegrowthFreq = (0.3,)

            else:
                raise ValueError(
                    "Moleules MC move ratios not listed for this solvent and solute or ensemble "
                    "in the GOMC control file writer."
                )

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


        gomc_control.write_gomc_control_file(
            gomc_charmm_object_with_files,
            output_name_control_file_name,
            used_ensemble,
            MC_steps,
            production_temperature_K,
            ff_psf_pdb_file_directory=None,
            check_input_files_exist=False,
            Parameters="{}.inp".format(gomc_ff_filename_str),
            Restart=True,
            Checkpoint=False,
            ExpertMode=False,
            Coordinates_box_0=Coordinates_box_0,
            Structure_box_0=Structure_box_0,
            binCoordinates_box_0=binCoordinates_box_0,
            extendedSystem_box_0=extendedSystem_box_0,
            binVelocities_box_0=None,
            Coordinates_box_1=None,
            Structure_box_1=None,
            binCoordinates_box_1=None,
            extendedSystem_box_1=None,
            binVelocities_box_1=None,
            input_variables_dict={
                "PRNG": seed_no,
                "Pressure": production_pressure_bar,
                "Ewald": use_ElectroStatics,
                "ElectroStatic": use_ElectroStatics,
                "VDWGeometricSigma": VDWGeometricSigma,
                "Rcut": (job.doc.Rcut_ang * u.angstrom).to("angstrom"),
                "Exclude": Exclude,
                "VolFreq": VolFreq[-1],
                "MultiParticleFreq": MultiParticleFreq[-1],
                "IntraSwapFreq": IntraSwapFreq[-1],
                "CrankShaftFreq": CrankShaftFreq[-1],
                "SwapFreq": SwapFreq[-1],
                "DisFreq": DisFreq[-1],
                "RotFreq": RotFreq[-1],
                "RegrowthFreq": RegrowthFreq[-1],
                "OutputName": output_name_control_file_name,
                "EqSteps": EqSteps,
                "PressureCalc": output_false_list_input,
                "RestartFreq": output_true_list_input,
                "CheckpointFreq": output_true_list_input,
                "ConsoleFreq": output_true_list_input,
                "BlockAverageFreq": output_true_list_input,
                "HistogramFreq": output_false_list_input,
                "CoordinatesFreq": output_false_list_input,
                "DCDFreq": output_true_list_input,
                "Potential": cutoff_style,
                "LRC": True,
                "RcutLow": 0 * u.angstrom,
                "CBMC_First": 12,
                "CBMC_Nth": 10,
                "CBMC_Ang": 50,
                "CBMC_Dih": 50,
                "FreeEnergyCalc": FreeEnergyCalc,
                "MoleculeType": MoleculeType,
                "InitialState": initial_state_sims_i,
                "LambdaVDW": list(job.doc.LambdaVDW_list),
                #"LambdaCoulomb": None,
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
# namd_equilb_NPT -starting the NAMD simulations (start)
# ******************************************************
# ******************************************************
@Project.pre(mosdef_input_written)
@Project.pre(part_2a_namd_equilb_NPT_control_file_written)
@Project.post(part_3a_output_namd_equilb_NPT_started)
@Project.post(part_4a_job_namd_equilb_NPT_completed_properly)
@Project.operation.with_directives(
    {
        "np": lambda job: job.doc.namd_node_ncpu,
        "ngpu": lambda job: job.doc.namd_node_ngpu,
        "memory": memory_needed,
        "walltime": walltime_namd_hr,
    }
)
@flow.with_job
@flow.cmd
def run_namd_equilb_NPT_gomc_command(job):
    """Run the namd_equilb_NPT simulation."""
    print("#**********************")
    print("# Started the run_namd_equilb_NPT_gomc_command.")
    print("#**********************")

    control_file_name_str = namd_equilb_NPT_control_file_name_str

    print(f"Running simulation job id {job}")
    run_command = "{}/{} +p{} {}.conf > out_{}.dat".format(
        str(namd_binary_path),
        str(job.doc.namd_equilb_NPT_gomc_binary_file),
        str(job.doc.namd_node_ncpu),
        str(control_file_name_str),
        str(control_file_name_str),
    )

    print('namd run_command = ' + str(run_command))

    return run_command


# ******************************************************
# ******************************************************
# namd_equilb_NPT -starting the NAMD simulations (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# equilb NPT - starting the GOMC simulation (start)
# ******************************************************
# ******************************************************

for initial_state_j in range(0, number_of_lambda_spacing_including_zero_int):
    @Project.pre(part_2a_namd_equilb_NPT_control_file_written)
    @Project.pre(part_4a_job_namd_equilb_NPT_completed_properly)
    @Project.post(part_3b_output_gomc_equilb_design_ensemble_started)
    @Project.post(part_4b_job_gomc_equilb_design_ensemble_completed_properly)
    @Project.operation.with_directives(
        {
            "np": lambda job: job.doc.gomc_ncpu,
            "ngpu": lambda job: job.doc.gomc_ngpu,
            "memory": memory_needed,
            "walltime": walltime_gomc_equilbrium_hr,
        },
        name = f"gomc_equilb_design_ensemble_initial_state_{initial_state_j}"
    )
    @flow.with_job
    @flow.cmd
    def run_equilb_run_gomc_command(job, *, initial_state_j=initial_state_j):
        """Run the gomc_equilb_run_ensemble simulation."""
        control_file_name_str = job.doc.gomc_equilb_design_ensemble_dict[
            str(initial_state_j)
        ]["output_name_control_file_name"]

        print(f"Running simulation job id {job}")
        run_command = "{}/{} +p{} {}.conf > out_{}.dat".format(
            str(gomc_binary_path),
            str(job.doc.gomc_equilb_design_ensemble_gomc_binary_file),
            str(job.doc.gomc_ncpu),
            str(control_file_name_str),
            str(control_file_name_str),
        )

        print('gomc equilbrium_run run_command = ' + str(run_command))

        return run_command
# *****************************************
# ******************************************************
# equilb NPT - starting the GOMC simulation (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# production run - starting the GOMC simulation (start)
# ******************************************************
# ******************************************************
for initial_state_i in range(0, number_of_lambda_spacing_including_zero_int):
    @Project.pre(part_2c_gomc_production_control_file_written)
    @Project.pre(part_4b_job_gomc_equilb_design_ensemble_completed_properly)
    @Project.post(part_part_3c_output_gomc_production_run_started)
    @Project.post(part_4c_job_production_run_completed_properly)
    @Project.operation.with_directives(
        {
            "np": lambda job: job.doc.gomc_ncpu,
            "ngpu": lambda job: job.doc.gomc_ngpu,
            "memory": memory_needed,
            "walltime": walltime_gomc_production_hr,
        },
        name = f"gomc_production_ensemble_initial_state_{initial_state_i}"
    )
    @flow.with_job
    @flow.cmd
    def run_production_run_gomc_command(job, *, initial_state_i=initial_state_i):
        """Run the gomc_production_ensemble simulation."""

        control_file_name_str = job.doc.gomc_production_run_ensemble_dict[
            str(initial_state_i)
        ]["output_name_control_file_name"]

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
# data analysis - get the average data from each individual simulation (start)
# ******************************************************
# ******************************************************

@Project.operation.with_directives(
     {
         "np": 1,
         "ngpu": 0,
         "memory": memory_needed,
         "walltime": walltime_gomc_analysis_hr,
     }
)
@FlowProject.pre(
     lambda *jobs: all(
         part_4c_job_production_run_completed_properly(job)
         for job in jobs
     )
)
@Project.pre(part_4c_job_production_run_completed_properly)
@Project.post(part_5a_analysis_individual_simulation_averages_completed)
@flow.with_job
def part_5a_analysis_individual_simulation_averages(*jobs):
    # remove the total averaged replicate data and all analysis data after this,
    # as it is no longer valid when adding more simulations
    if os.path.isfile(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_box_0}'):
        os.remove(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_box_0}')

    output_column_temp_title = 'temp_K'  # column title title for temp
    output_column_solute_title = 'solute'  # column title title for temp
    output_column_dFE_MBAR_title = 'dFE_MBAR_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_MBAR_std_title = 'dFE_MBAR_std_kcal_per_mol'  # column title title for ds_MBAR
    output_column_dFE_TI_title = 'dFE_TI_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_TI_std_title = 'dFE_TI_std_kcal_per_mol'  # column title title for ds_MBAR
    output_column_dFE_BAR_title = 'dFE_BAR_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_BAR_std_title = 'dFE_BAR_std_kcal_per_mol'  # column title title for ds_MBAR


    # get the averages from each individual simulation and write the csv's.
    for job in jobs:
        files = []
        k_b = 1.9872036E-3  # kcal/mol/K
        temperature = job.sp.production_temperature_K
        k_b_T = temperature * k_b

        for initial_state_iter in range(0, number_of_lambda_spacing_including_zero_int):
            reading_filename_box_0_iter = f'Free_Energy_BOX_0_{gomc_production_control_file_name_str}_' \
                                          f'initial_state_{initial_state_iter}.dat'
            files.append(reading_filename_box_0_iter)

        # for TI estimator
        dHdl = pd.concat([extract_dHdl(job.fn(f), T=temperature) for f in files])
        ti = TI().fit(dHdl)
        delta_ti, delta_std_ti = get_delta_TI_or_MBAR(ti, k_b_T)

        # for MBAR estimator
        u_nk = pd.concat([extract_u_nk(job.fn(f), T=temperature) for f in files])
        mbar = MBAR().fit(u_nk)
        delta_mbar, delta_std_mbar = get_delta_TI_or_MBAR(mbar, k_b_T)

        # for BAR estimator
        bar = BAR().fit(u_nk)
        delta_bar, delta_std_bar = get_delta_BAR(bar, k_b_T)

        # write the data out in each job
        box_0_replicate_data_txt_file = open(job.fn(output_replicate_txt_file_name_box_0), "w")
        box_0_replicate_data_txt_file.write(
            f"{output_column_temp_title: <30} "
            f"{output_column_solute_title: <30} "
            f"{output_column_dFE_MBAR_title: <30} "
            f"{output_column_dFE_MBAR_std_title: <30} "
            f"{output_column_dFE_TI_title: <30} "
            f"{output_column_dFE_TI_std_title: <30} "
            f"{output_column_dFE_BAR_title: <30} "
            f"{output_column_dFE_BAR_std_title: <30} "
            f" \n"
        )
        box_0_replicate_data_txt_file.write(
            f"{job.sp.production_temperature_K: <30} "
            f"{job.sp.solute: <30} "
            f"{delta_mbar: <30} "
            f"{delta_std_mbar: <30} "
            f"{delta_ti: <30} "
            f"{delta_std_ti: <30} "
            f"{delta_bar: <30} "
            f"{delta_std_bar: <30} "
            f" \n"
        )


# ******************************************************
# ******************************************************
# data analysis - get the average data from each individual simulation (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# data analysis - get the average and std. dev. from/across all the replicates (start)
# ******************************************************
# ******************************************************

@aggregator.groupby(key=statepoint_without_replica,
                    sort_by="production_temperature_K",
                    sort_ascending=True
)
@Project.operation.with_directives(
     {
         "np": 1,
         "ngpu": 0,
         "memory": memory_needed,
         "walltime": walltime_gomc_analysis_hr,
     }
)

@Project.pre(lambda *jobs: all(part_5a_analysis_individual_simulation_averages_completed(j)
                               for j in jobs[0]._project))
@Project.pre(part_4c_job_production_run_completed_properly)
@Project.pre(part_5a_analysis_individual_simulation_averages_completed)
@Project.post(part_5b_analysis_replica_averages_completed)
def part_5b_analysis_replica_averages(*jobs):
    # ***************************************************
    #  create the required lists and file labels for the replicates (start)
    # ***************************************************
    # output and labels
    output_column_temp_title = 'temp_K'  # column title title for temp
    output_column_temp_std_title = 'temp_std_K'  # column title title for temp
    output_column_solute_title = 'solute'  # column title title for temp
    output_column_dFE_MBAR_title = 'dFE_MBAR_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_MBAR_std_title = 'dFE_MBAR_std_kcal_per_mol'  # column title title for ds_MBAR
    output_column_dFE_TI_title = 'dFE_TI_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_TI_std_title = 'dFE_TI_std_kcal_per_mol'  # column title title for ds_MBAR
    output_column_dFE_BAR_title = 'dFE_BAR_kcal_per_mol'  # column title title for delta_MBAR
    output_column_dFE_BAR_std_title = 'dFE_BAR_std_kcal_per_mol'  # column title title for ds_MBAR

    # get the list used in this function
    temp_repilcate_list = []
    solute_repilcate_list = []

    delta_MBAR_repilcate_box_0_list = []
    delta_TI_repilcate_box_0_list = []
    delta_BAR_repilcate_box_0_list = []


    output_txt_file_header = f"{output_column_temp_title: <30} " \
                             f"{output_column_temp_std_title: <30} " \
                             f"{output_column_solute_title: <30} "\
                             f"{output_column_dFE_MBAR_title: <30} "\
                             f"{output_column_dFE_MBAR_std_title: <30} "\
                             f"{output_column_dFE_TI_title: <30} "\
                             f"{output_column_dFE_TI_std_title: <30} "\
                             f"{output_column_dFE_BAR_title: <30} "\
                             f"{output_column_dFE_BAR_std_title: <30} "\
                             f"\n"


    write_file_path_and_name_box_0 = f'analysis/{output_avg_std_of_replicates_txt_file_name_box_0}'
    if os.path.isfile(write_file_path_and_name_box_0):
        box_box_0_data_txt_file = open(write_file_path_and_name_box_0, "a")
    else:
        box_box_0_data_txt_file = open(write_file_path_and_name_box_0, "w")
        box_box_0_data_txt_file.write(output_txt_file_header)


    # ***************************************************
    #  create the required lists and file labels for the replicates (end)
    # ***************************************************

    for job in jobs:

        # *************************
        # drawing in data from single file and extracting specific rows from box 0 (start)
        # *************************
        reading_file_box_box_0 = job.fn(output_replicate_txt_file_name_box_0)

        data_box_box_0 = pd.read_csv(reading_file_box_box_0, sep='\s+', header=0, na_values='NaN', index_col=False)
        data_box_box_0 = pd.DataFrame(data_box_box_0)

        temp_repilcate_list.append(data_box_box_0.loc[:, output_column_temp_title][0])
        solute_repilcate_list.append(data_box_box_0.loc[:, output_column_solute_title][0])

        delta_MBAR_repilcate_box_0_list.append(data_box_box_0.loc[:, output_column_dFE_MBAR_title][0])
        delta_TI_repilcate_box_0_list.append(data_box_box_0.loc[:, output_column_dFE_TI_title][0])
        delta_BAR_repilcate_box_0_list.append(data_box_box_0.loc[:, output_column_dFE_BAR_title][0])

        # *************************
        # drawing in data from single file and extracting specific rows from box 0 (end)
        # *************************


    # *************************
    # get the replica means and std.devs (start)
    # *************************
    temp_mean = np.mean(temp_repilcate_list)
    temp_std = np.std(temp_repilcate_list, ddof=1)

    solute_iter = solute_repilcate_list[0]

    delta_MBAR_mean_box_box_0 = np.mean(delta_MBAR_repilcate_box_0_list)
    delta_TI_mean_box_box_0 = np.mean(delta_TI_repilcate_box_0_list)
    delta_BAR_mean_box_box_0 = np.mean(delta_BAR_repilcate_box_0_list)

    delta_std_MBAR_mean_box_box_0 = np.std(delta_MBAR_repilcate_box_0_list, ddof=1)
    delta_std_TI_mean_box_box_0 = np.std(delta_TI_repilcate_box_0_list, ddof=1)
    delta_std_BAR_mean_box_box_0 = np.std(delta_BAR_repilcate_box_0_list, ddof=1)

    # *************************
    # get the replica means and std.devs (end)
    # *************************

    # ************************************
    # write the analysis data files for the liquid and vapor boxes (start)
    # ************************************

    box_box_0_data_txt_file.write(
        f"{temp_mean: <30} "
        f"{temp_std: <30} "
        f"{solute_iter: <30} "
        f"{delta_MBAR_mean_box_box_0: <30} "
        f"{delta_std_MBAR_mean_box_box_0: <30} "
        f"{delta_TI_mean_box_box_0: <30} "
        f"{delta_std_TI_mean_box_box_0: <30} "
        f"{delta_BAR_mean_box_box_0: <30} "
        f"{delta_std_BAR_mean_box_box_0: <30} "
        f" \n"
    )

    # ************************************
    # write the analysis data files for the liquid and vapor boxes (end)
    # ************************************


# ******************************************************
# ******************************************************
# data analysis - get the average and std. dev. from/across all the replicates (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# signac end code (start)
# ******************************************************
# ******************************************************
if __name__ == "__main__":
    pr = Project()
    pr.main()
# ******************************************************
# ******************************************************
# signac end code (end)
# ******************************************************
# ******************************************************
