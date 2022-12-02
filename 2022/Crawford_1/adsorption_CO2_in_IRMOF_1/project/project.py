"""GOMC's setup for signac, signac-flow, signac-dashboard for this study."""
# project.py

import flow

from mbuild.lattice import load_cif
from mbuild.utils.io import get_fn
import mbuild as mb
import mosdef_gomc.formats.gmso_charmm_writer as mf_charmm
import mosdef_gomc.formats.gmso_gomc_conf_writer as gomc_control
from gmso import Topology
from gmso.external.convert_mbuild import to_mbuild
import pandas as pd
import numpy as np
import os

import unyt as u
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment



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

# Enter the GOMC binary path here (MANDATORY INFORMAION)
gomc_binary_path = "/home/brad/Programs/GOMC/GOMC_2_76/bin"

# number of steps
gomc_steps_equilb_design_ensemble = 5 * 10**6 # set value for paper = 5 * 10**6
gomc_steps_production = 30 * 10**6 # set value for paper = 10 * 10**6

gomc_output_data_every_X_steps = 100 * 10**3 # set value for paper = 100 * 10**3

# force field (FF) file for all simulations in that job
# Note: do not add extensions
gomc_ff_filename_str = "in_gomc_FF"

# initial mosdef structure and coordinates
# Note: do not add extensions
mosdef_structure_box_0_name_str = "mosdef_box_0"
mosdef_structure_box_1_name_str = "mosdef_box_1"

# The equilb run using the ensemble used for the simulation design, which
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

# Analysis (molecules / zeolite unit cell):
# Output text (txt) file name for the molecules / zeolite unit cell values of the system
# using the averages and std. devs. all the replcates, including the extention (.txt, .dat, etc)
output_molecules_per_zeolite_unit_cell_avg_std_txt_file_name = \
    "analysis_molecules_per_zeolite_unit_cell_avg_std_of_replicates.txt"

walltime_mosdef_hr = 24
walltime_gomc_equilbrium_hr = 128
walltime_gomc_production_hr = 368
walltime_gomc_analysis_hr = 4
memory_needed = 16

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

    # zeolite unit cells  (NOTE THIS IS FIXED IN via the read in mol2 files )
    # these are only used for the analysis calcs and the mol2 file is a 2x2x2 UC zeolite.
    job.doc.No_zeolite_unit_cell_x_axis = 2
    job.doc.No_zeolite_unit_cell_y_axis = 2
    job.doc.No_zeolite_unit_cell_z_axis = 2

    # gomc core and CPU or GPU
    job.doc.gomc_ncpu = 1  # 1 is optimal
    job.doc.gomc_ngpu = 0

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

    # set the ensemble type
    job.doc.gomc_equilb_design_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_GCMC"
    job.doc.gomc_production_ensemble_gomc_binary_file = f"GOMC_{job.doc.gomc_cpu_or_gpu}_GCMC"

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
# @Project.pre(select_production_ensemble)
@Project.label
@flow.with_job
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
# check if GOMC psf, pdb, and force field (FF) files were written (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC control files were written (start)
# ******************************************************
# ******************************************************

def gomc_control_file_written(job, control_filename_str):
    """Check that the gomc control files are written."""
    file_written_bool = False
    control_file = f"{control_filename_str}.conf"

    if job.isfile(control_file):
        with open(f"{control_file}", "r") as fp:
            out_gomc = fp.readlines()
            for i, line in enumerate(out_gomc):
                if "OutputName" in line:
                    split_move_line = line.split()
                    if split_move_line[0] == "OutputName":
                        file_written_bool = True

    return file_written_bool


@Project.label
@flow.with_job
def part_2a_gomc_equilb_design_ensemble_control_file_written(job):
    """Check that the gomc_equilb_design_ensemble control file is written (run temperature)."""
    return gomc_control_file_written(job, gomc_equilb_design_ensemble_control_file_name_str)


# checking if the GOMC control file is written for the production run
@Project.label
@flow.with_job
def part_2b_production_control_file_written(job):
    """Check that the gomc production run control file is written (run temperature)."""
    return gomc_control_file_written(job, gomc_production_control_file_name_str)

# ******************************************************
# ******************************************************
# check if GOMC control files were written (end)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# check if GOMC simulations started (start)
# ******************************************************
# ******************************************************

def gomc_simulation_started(job, control_filename_str):
    """General check to see if the gomc simulation is started."""
    output_started_bool = False
    if job.isfile("out_{}.dat".format(control_filename_str)) and job.isfile(
        "{}_merged.psf".format(control_filename_str)
    ):
        output_started_bool = True

    return output_started_bool

@Project.label
@flow.with_job
def part_3a_output_gomc_equilb_design_ensemble_started(job):
    """Check if the gomc_equilb_design_ensemble simulation is started (set temperature)."""
    return gomc_simulation_started(job, gomc_equilb_design_ensemble_control_file_name_str)

# check if production GOMC run is started by seeing if the GOMC consol file and the merged psf exist
@Project.label
@flow.with_job
def part_3b_output_gomc_production_run_started(job):
    """Check to see if the gomc_production_run simulation is started (set temperature)."""
    return gomc_simulation_started(job, gomc_production_control_file_name_str)
# ******************************************************
# ******************************************************
# check if GOMC simulations started (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# check if GOMC simulations are completed properly (start)
# ******************************************************
# ******************************************************

def gomc_sim_completed_properly(job, control_filename_str):
    """Check to see if the gomc simulation was completed properly."""
    job_run_properly_bool = False
    output_log_file = "out_{}.dat".format(control_filename_str)
    if job.isfile(output_log_file):
        # with open(f"workspace/{job.id}/{output_log_file}", "r") as fp:
        with open(f"{output_log_file}", "r") as fp:
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

@Project.label
@flow.with_job
def part_4a_job_gomc_equilb_design_ensemble_completed_properly(job):
    """Check if the gomc_equilb_design_ensemble simulation was completed properly (set temperature)."""
    return gomc_sim_completed_properly(job, gomc_equilb_design_ensemble_control_file_name_str)

# check if production GOMC run completed by checking the end of the GOMC consol file
@Project.label
@flow.with_job
def part_4b_job_gomc_production_run_completed_properly(job):
    """Check to see if the gomc_production_run simulation was completed properly (set temperature)."""
    return gomc_sim_completed_properly(job, gomc_production_control_file_name_str)

# ******************************************************
# ******************************************************
# check if GOMC simulations are completed properly (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# check if GOMC simulations analysis is completed (start)
# ******************************************************
# ******************************************************

# check if analysis is done for the individual replicates wrote the gomc files
@Project.pre(part_4b_job_gomc_production_run_completed_properly)
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
    """Check that the individual simulation averages files are written ."""
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
# check if GOMC simulations analysis is completed (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# build the system, with option to write the force field (force field (FF)), pdb, psf files.
# Note: this is needed to write GOMC control file, even if a restart (start)
# ******************************************************

def build_charmm(job, write_files=True):
    """Build the Charmm object and potentially write the pdb, psd, and force field (FF) files."""
    print("#**********************")
    print("Started: GOMC Charmm Object")
    print("#**********************")
    mbuild_box_seed_no = job.doc.replica_number_int

    # load and build the ETV with 3 cell units in every direction (x, y, and z)
    lattice_mol2_zeolite_cell = Topology.load('../../src/molecules/2x2x2_UC_IRMOF_1.mol2')
    lattice_mol2_zeolite_cell = to_mbuild(lattice_mol2_zeolite_cell)
    lattice_mol2_zeolite_cell.box = mb.Box(lengths=[5.1665, 5.1665, 5.1665],
                                           angles=[90.00, 90.00, 90.00],
                                           )
    lattice_mol2_zeolite_cell.name = 'MOF'
    lattice_mol2_zeolite_cell_FF_file = "../../src/xmls/DREIDING_IRMOF_1.xml"

    if job.sp.molecule == 'CO2':
        adsorbate_molecule_A = Topology.load('../../src/molecules/CO2.mol2')
        adsorbate_molecule_A = to_mbuild(adsorbate_molecule_A)
        adsorbate_molecule_A.name = 'CO2'
        adsorbate_molecule_A_FF_file = "../../src/xmls/trappe_CO2.xml"

    else:
        raise ValueError("ERROR: the only valid molecule entries is 'CO2'")

    FF_file_dict = {
        '_C1': lattice_mol2_zeolite_cell_FF_file,
        '_C2': lattice_mol2_zeolite_cell_FF_file,
        '_C3': lattice_mol2_zeolite_cell_FF_file,
        '_H3': lattice_mol2_zeolite_cell_FF_file,
        '_ZN1': lattice_mol2_zeolite_cell_FF_file,
        '_O1': lattice_mol2_zeolite_cell_FF_file,
        '_O2': lattice_mol2_zeolite_cell_FF_file,
        adsorbate_molecule_A.name: adsorbate_molecule_A_FF_file
    }

    fix_bonds_angles_list = [adsorbate_molecule_A.name]

    fix_atoms_list = ['_C1', '_C2', '_C3', '_H3', '_ZN1','_O1', '_O2']
    residues_list = ['_C1', '_C2', '_C3', '_H3', '_ZN1','_O1', '_O2', adsorbate_molecule_A.name]

    bead_to_atom_name_dict = {'_C1': 'C1', '_C2': 'C2', '_C3': 'C3',
                              '_H3': 'H3',
                              '_ZN1': 'ZN',
                              '_O1': 'O1', '_O2': 'O2'}

    print('Running: vapor phase box packing')
    box_vap = mb.fill_box(compound=adsorbate_molecule_A,
                          n_compounds=2000,
                          box=[10.0, 10.0, 10.0],
                          seed=mbuild_box_seed_no
                          )
    print('Completed: vapor phase box packing')

    print('Running: GOMC FF file, and the psf and pdb files')
    gomc_charmm = mf_charmm.Charmm(
        lattice_mol2_zeolite_cell,
        mosdef_structure_box_0_name_str,
        structure_box_1=box_vap,
        filename_box_1=mosdef_structure_box_1_name_str,
        ff_filename=gomc_ff_filename_str,
        forcefield_selection=FF_file_dict,
        residues=residues_list,
        bead_to_atom_name_dict=bead_to_atom_name_dict,
        gomc_fix_bonds_angles=fix_bonds_angles_list,
        fix_residue=fix_atoms_list,
        gmso_match_ff_by="molecule",
    )

    if write_files == True:
        gomc_charmm.write_inp()

        gomc_charmm.write_psf()

        gomc_charmm.write_pdb()

    print('Completed: GOMC FF file, and the psf and pdb files')

    return gomc_charmm


# ******************************************************
# ******************************************************
# build the system, with option to write the force field (force field (FF)), pdb, psf files.
# Note: this is needed to write GOMC control file, even if a restart (end)
# ******************************************************


# ******************************************************
# ******************************************************
# Creating GOMC files (pdb, psf, force field (FF), and gomc control files (start)
# ******************************************************
# ******************************************************
@Project.pre(part_1a_initial_data_input_to_json)
@Project.post(part_2a_gomc_equilb_design_ensemble_control_file_written)
@Project.post(part_2b_production_control_file_written)
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
    """Build the Charmm object and write the pdb, psd, and force field (FF) files
    for all the simulations in the workspace."""
    gomc_charmm_object_with_files = build_charmm(job, write_files=True)

    # ******************************************************
    # common variables (start)
    # ******************************************************
    # variables from signac workspace
    production_temperature_K = job.sp.production_temperature_K * u.K
    production_pressure_bar = job.sp.production_pressure_bar * u.bar
    production_fugacity_bar = job.sp.production_fugacity_bar * u.bar
    zeolite_fugacity_bar = 0 * u.bar
    seed_no = job.doc.replica_number_int

    # cutoff, tail correction, and 1-4 interactions
    Rcut_ang = 12.8 * u.angstrom
    RcutCoulomb_box_0 = 12.8 * u.angstrom
    LRC = False
    Exclude = "1-4"

    # MC move ratios
    DisFreq = 0.2
    VolFreq = 0.0
    RotFreq = 0.1
    RegrowthFreq = 0.0
    IntraSwapFreq = 0.10
    SwapFreq = 0.6

    Fugacity_dict = {
        '_C1': zeolite_fugacity_bar,
        '_C2': zeolite_fugacity_bar,
        '_C3': zeolite_fugacity_bar,
        '_H3': zeolite_fugacity_bar,
        '_ZN1': zeolite_fugacity_bar,
        '_O1': zeolite_fugacity_bar,
        '_O2': zeolite_fugacity_bar,
        "CO2": production_fugacity_bar
    }

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
    # common variables (end)
    # ******************************************************

    # ******************************************************
    # equilibrium GCMC- GOMC control file writing  (start)
    # ******************************************************

    print("#**********************")
    print("Started: equilb GCMC GOMC control file writing")
    print("#**********************")

    output_name_control_file_name = gomc_equilb_design_ensemble_control_file_name_str
    starting_control_file_name_str = gomc_charmm_object_with_files

    # calc MC steps for gomc equilb
    MC_steps = int(gomc_steps_equilb_design_ensemble)
    EqSteps = 1000

    # write the control file
    gomc_control.write_gomc_control_file(
        starting_control_file_name_str,
        output_name_control_file_name,
        'GCMC',
        MC_steps,
        production_temperature_K,
        ff_psf_pdb_file_directory=None,
        check_input_files_exist=False,
        Parameters=None,
        Restart=False,
        Checkpoint=False,
        ExpertMode=False,
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
            "Ewald": True,
            "ElectroStatic": True,
            "VDWGeometricSigma": False,
            "Rcut": Rcut_ang,
            "RcutCoulomb_box_0": RcutCoulomb_box_0,
            "LRC": LRC,
            "Fugacity": Fugacity_dict,
            "Exclude": Exclude,
            "DisFreq": DisFreq,
            "VolFreq": VolFreq,
            "RotFreq": RotFreq,
            "RegrowthFreq": RegrowthFreq,
            "IntraSwapFreq": IntraSwapFreq,
            "SwapFreq": SwapFreq,
            "OutputName": output_name_control_file_name,
            "EqSteps": EqSteps,
            "PressureCalc": output_true_list_input,
            "RestartFreq": output_true_list_input,
            "CheckpointFreq": output_true_list_input,
            "ConsoleFreq": output_true_list_input,
            "BlockAverageFreq": output_true_list_input,
            "HistogramFreq": output_false_list_input,
            "CoordinatesFreq": output_false_list_input,
            "DCDFreq": output_true_list_input,
            "Potential": "VDW",
            "RcutLow": 0.0 * u.angstrom,
            "CBMC_First": 10,
            "CBMC_Nth": 8,
            "CBMC_Ang": 5,
            "CBMC_Dih": 5,
            "SampleFreq": 200,
            "MEMC_DataInput": None,
        },
    )
    print("#**********************")
    print("Completed: equilb GCMC GOMC control file written")
    print("#**********************")

    # ******************************************************
    # equilibrium GCMC - GOMC control file writing  (end)
    # ******************************************************

    # ******************************************************
    # production GCMC - GOMC control file writing (start)
    # ******************************************************

    print("#**********************")
    print("Started: production GCMC GOMC control file writing")
    print("#**********************")

    output_name_control_file_name = gomc_production_control_file_name_str
    restart_control_file_name_str = gomc_equilb_design_ensemble_control_file_name_str

    # calc MC steps
    MC_steps = int(gomc_steps_production)
    EqSteps = 1000

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

    # write control file
    gomc_control.write_gomc_control_file(
        gomc_charmm_object_with_files,
        output_name_control_file_name,
        "GCMC",
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
            "Ewald": True,
            "ElectroStatic": True,
            "VDWGeometricSigma": False,
            "Rcut": Rcut_ang,
            "RcutCoulomb_box_0": RcutCoulomb_box_0,
            "LRC": LRC,
            "Fugacity": Fugacity_dict,
            "Exclude": Exclude,
            "DisFreq": DisFreq,
            "VolFreq": VolFreq,
            "RotFreq": RotFreq,
            "RegrowthFreq": RegrowthFreq,
            "IntraSwapFreq": IntraSwapFreq,
            "SwapFreq": SwapFreq,
            "OutputName": output_name_control_file_name,
            "EqSteps": EqSteps,
            "PressureCalc": output_true_list_input,
            "RestartFreq": output_true_list_input,
            "CheckpointFreq": output_true_list_input,
            "ConsoleFreq": output_true_list_input,
            "BlockAverageFreq": output_true_list_input,
            "HistogramFreq": output_false_list_input,
            "CoordinatesFreq": output_false_list_input,
            "DCDFreq": output_true_list_input,
            "Potential": "VDW",
            "RcutLow": 0.0 * u.angstrom,
            "CBMC_First": 12,
            "CBMC_Nth": 10,
            "CBMC_Ang": 50,
            "CBMC_Dih": 50,
            "SampleFreq": 200,
            "MEMC_DataInput": None,
        },
    )

    print("#**********************")
    print("Completed: production GCMC GOMC control file writing")
    print("#**********************")
    # ******************************************************
    # production GCMC - GOMC control file writing(end)
    # ******************************************************


# ******************************************************
# ******************************************************
# Creating GOMC files (pdb, psf, force field (FF), and gomc control files (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# equilb GCMC- starting the GOMC simulation (start)
# ******************************************************
# ******************************************************

@Project.pre(mosdef_input_written)
@Project.pre(part_2a_gomc_equilb_design_ensemble_control_file_written)
@Project.post(part_3a_output_gomc_equilb_design_ensemble_started)
@Project.post(part_4a_job_gomc_equilb_design_ensemble_completed_properly)
@Project.operation.with_directives(
    {
        "np": lambda job: job.doc.gomc_ncpu,
        "ngpu": lambda job: job.doc.gomc_ngpu,
        "memory": memory_needed,
        "walltime": walltime_gomc_equilbrium_hr,
    },
)
@flow.with_job
@flow.cmd
def run_gomc_equilb_ensemble_command(job):
    """Run the gomc_equilb_ensemble simulation."""
    control_file_name_str = gomc_equilb_design_ensemble_control_file_name_str

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
# equilb GCMC - starting the GOMC simulation (end)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# production run - starting the GOMC simulation (start)
# ******************************************************
# ******************************************************


@Project.pre(part_2b_production_control_file_written)
@Project.pre(part_4a_job_gomc_equilb_design_ensemble_completed_properly)
@Project.post(part_3b_output_gomc_production_run_started)
@Project.post(part_4b_job_gomc_production_run_completed_properly)
@Project.operation.with_directives(
    {
        "np": lambda job: job.doc.gomc_ncpu,
        "ngpu": lambda job: job.doc.gomc_ngpu,
        "memory": memory_needed,
        "walltime": walltime_gomc_production_hr,
    },
)
@flow.with_job
@flow.cmd
def run_gomc_production_run_command(job):
    """Run the gomc_production_run simulation."""

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


@Project.operation.with_directives(
     {
         "np": 1,
         "ngpu": 0,
         "memory": memory_needed,
         "walltime": walltime_gomc_analysis_hr,
     }
)
@FlowProject.pre(
     lambda * jobs: all(
         part_4b_job_gomc_production_run_completed_properly(job, gomc_production_control_file_name_str)
         for job in jobs
     )
)
@Project.pre(part_4b_job_gomc_production_run_completed_properly)
@Project.post(part_5a_analysis_individual_simulation_averages_completed)
@flow.with_job
def part_5a_analysis_individual_simulation_averages(*jobs):
    # remove the total averged replicate data and all analysis data after this,
    # as it is no longer valid when adding more simulations
    if os.path.isfile(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_box_0}'):
        os.remove(f'../../analysis/{output_avg_std_of_replicates_txt_file_name_box_0}')
    if os.path.isfile(f"../../analysis/{output_molecules_per_zeolite_unit_cell_avg_std_txt_file_name}"):
        os.remove(f"../../analysis/{output_molecules_per_zeolite_unit_cell_avg_std_txt_file_name}")


    # this is set to basically use all values.  However, allows the ability to set if needed
    step_start = 0 * 10 ** 6
    step_finish = 1 * 10 ** 12

    # get the averages from each individual simulation and write the csv's.
    for job in jobs:

        reading_file_box_0 = job.fn(f'Blk_{gomc_production_control_file_name_str}_BOX_0.dat')
        reading_file_box_1 = job.fn(f'Blk_{gomc_production_control_file_name_str}_BOX_1.dat')

        output_column_temp_title = 'temp_K'  # column title for temp
        output_column_molecule_name_title = 'molecule_name'  # column title for molecule name value
        output_column_no_step_title = 'Step'  # column title for iter value
        output_column_total_molecules_title = "No_mol"  # column title for TOT_MOL
        output_column_fraction_adsorbed_molecules_title = "adsorbed_mol_fraction"  # column title for TOT_MOL
        output_column_Rho_title = 'Rho_kg_per_m_cubed'  # column title for TOT_DENS
        output_column_fraction_adsorbed_Rho_title = "adsorbed_Rho_fraction"  # column title for TOT_MOL
        output_column_pressure_title = 'P_bar'  # column title title for PRESSURE

        blk_file_reading_column_no_step_title = '#STEP'  # column title for iter value
        blk_file_reading_column_total_molecules_title = "TOT_MOL"  # column title for TOT_MOL
        blk_file_reading_column_fraction_molecules_CO2_title = "MOLFRACT_CO2"  # column title for MOLFRACT_CO2
        blk_file_reading_column_Rho_title = 'TOT_DENS'  # column title for TOT_DENS
        blk_file_reading_column_fraction_Rho_CO2_title = "MOLDENS_CO2"  # column title for MOLDENS_CO2


        # Programmed data
        step_start_string = str(int(step_start))
        step_finish_string = str(int(step_finish))

        # *************************
        # drawing in data from single file and extracting specific rows for box 0 /zeolite (start)
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

        total_molecules_box_0 = data_box_0.loc[:, blk_file_reading_column_total_molecules_title]
        total_molecules_box_0 = list(total_molecules_box_0)
        total_molecules_box_0 = np.transpose(total_molecules_box_0)
        total_molecules_box_0_mean = np.nanmean(total_molecules_box_0)

        Rho_box_0 = data_box_0.loc[:, blk_file_reading_column_Rho_title]
        Rho_box_0 = list(Rho_box_0)
        Rho_box_0 = np.transpose(Rho_box_0)
        Rho_box_0_mean = np.nanmean(Rho_box_0)

        if job.sp.molecule == "CO2":
            adsorbed_fraction_molecules_box_0 = \
                data_box_0.loc[:, blk_file_reading_column_fraction_molecules_CO2_title]
            adsorbed_fraction_Rho_box_0 = \
                data_box_0.loc[:, blk_file_reading_column_fraction_Rho_CO2_title]
        else:
            raise ValueError("ERROR: Only the CO2 analysis is supported in the current setup.")

        adsorbed_fraction_molecules_box_0 = list(adsorbed_fraction_molecules_box_0)
        adsorbed_fraction_molecules_box_0 = np.transpose(adsorbed_fraction_molecules_box_0)
        adsorbed_fraction_molecules_box_0_mean = np.nanmean(adsorbed_fraction_molecules_box_0)

        adsorbed_fraction_Rho_box_0 = list(adsorbed_fraction_Rho_box_0)
        adsorbed_fraction_Rho_box_0 = np.transpose(adsorbed_fraction_Rho_box_0)
        adsorbed_fraction_Rho_box_0_mean = np.nanmean(adsorbed_fraction_Rho_box_0)

        # *************************
        # drawing in data from single file and extracting specific rows for box 0 /zeolite (end)
        # *************************

        # ***********************
        # calc the (1) molecules / unit cell, (2) molecules / volume
        # (1) density / unit cell (g/mL), (2) density / volume (g) (start)
        # ***********************

        # ***********************
        # calc the avg data from the boxes (start)
        # ***********************

        # ***********************
        # calc the avg data from the boxes (start)
        # ***********************

        box_0_replicate_data_txt_file = open(output_replicate_txt_file_name_box_0, "w")
        box_0_replicate_data_txt_file.write(
            f"{output_column_temp_title: <30} "
            f"{output_column_pressure_title: <30} "
            f"{output_column_molecule_name_title: <30} "
            f"{output_column_total_molecules_title: <30} "
            f"{output_column_fraction_adsorbed_molecules_title: <30} "
            f"{output_column_Rho_title: <30} "
            f"{output_column_fraction_adsorbed_Rho_title: <30} "
            f" \n"
        )
        box_0_replicate_data_txt_file.write(
            f"{job.sp.production_temperature_K: <30} "
            f"{job.sp.production_pressure_bar: <30} "
            f"{job.sp.molecule: <30} "
            f"{total_molecules_box_0_mean: <30} "
            f"{adsorbed_fraction_molecules_box_0_mean: <30} "
            f"{Rho_box_0_mean: <30} "
            f"{adsorbed_fraction_Rho_box_0_mean: <30} "
            f" \n"
        )

        # ***********************
        # calc the avg data from the boxes (end)
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

@aggregator.groupby(key=statepoint_without_replica, sort_by="production_pressure_bar", sort_ascending=False)
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
@Project.pre(part_4b_job_gomc_production_run_completed_properly)
@Project.pre(part_5a_analysis_individual_simulation_averages_completed)
@Project.post(part_5b_analysis_replica_averages_completed)
def part_5b_analysis_replica_averages(*jobs):
    # ***************************************************
    #  create the required lists and file labels total averages across the replicates (start)
    # ***************************************************
    # get the list used in this function
    temp_repilcate_list = []
    pressure_repilcate_list = []

    total_molecules_repilcate_box_0_list = []
    adsorbed_fraction_molecules_box_0_list = []
    Rho_repilcate_box_0_list = []
    adsorbed_fraction_Rho_box_0_list = []
    molecule_name_list = []
    No_unit_cells_list = []

    output_column_temp_title = 'temp_K'
    output_column_pressure_title = 'P_bar'  # column title title for PRESSURE
    output_column_molecule_name_title = 'molecule_name'  # column title for molecule name value
    output_column_total_molecules_title = "No_mol"
    output_column_fraction_adsorbed_molecules_title = "adsorbed_mol_fraction"
    output_column_Rho_title = 'Rho_kg_per_m_cubed'
    output_column_fraction_adsorbed_Rho_title = "adsorbed_Rho_fraction"
    output_column_No_adsorbed_molecules_mean_title = 'No_adsorbed_molecules_mean'
    output_column_adsorbed_molecules_Rho_mean_title = "adsorbed_molecules_Rho_mean"
    output_column_adsorbed_molecule_per_unit_cell_mean_title = 'ads_molecules_per_UC_mean'

    output_column_temp_std_title = 'temp_std_K'
    output_column_pressure_std_title = 'P_std_bar'
    output_column_total_molecules_std_title = "No_mol_std"
    output_column_fraction_adsorbed_molecules_std_title = "adsorbed_mol_fraction_std"
    output_column_Rho_std_title = 'Rho_std_kg_per_m_cubed'
    output_column_fraction_adsorbed_Rho_std_title = "adsorbed_Rho_fraction_std"
    output_column_No_adsorbed_molecules_std_title = 'No_adsorbed_molecules_std'
    output_column_adsorbed_molecules_Rho_std_title = "adsorbed_molecules_Rho_std"
    output_column_adsorbed_molecule_per_unit_cell_std_title = 'ads_molecules_per_UC_std'


    output_txt_file_header = f"{output_column_temp_title: <30} " \
                             f"{output_column_temp_std_title: <30} " \
                             f"{output_column_pressure_title: <30} " \
                             f"{output_column_pressure_std_title: <30} " \
                             f"{output_column_molecule_name_title: <30} " \
                             f"{output_column_total_molecules_title: <30} " \
                             f"{output_column_total_molecules_std_title: <30} " \
                             f"{output_column_fraction_adsorbed_molecules_title: <30} " \
                             f"{output_column_fraction_adsorbed_molecules_std_title: <30} " \
                             f"{output_column_Rho_title: <30} "\
                             f"{output_column_Rho_std_title: <30} " \
                             f"{output_column_fraction_adsorbed_Rho_title: <30} " \
                             f"{output_column_fraction_adsorbed_Rho_std_title: <30} " \
                             f"{output_column_No_adsorbed_molecules_mean_title: <30} " \
                             f"{output_column_No_adsorbed_molecules_std_title: <30} " \
                             f"{output_column_adsorbed_molecules_Rho_mean_title: <30} " \
                             f"{output_column_adsorbed_molecules_Rho_std_title: <30} " \
                             f"{output_column_adsorbed_molecule_per_unit_cell_mean_title: <30} " \
                             f"{output_column_adsorbed_molecule_per_unit_cell_std_title: <30} " \
                             f"\n"


    write_file_path_and_name_box_0 = f'analysis/{output_avg_std_of_replicates_txt_file_name_box_0}'
    if os.path.isfile(write_file_path_and_name_box_0):
        box_0_data_txt_file = open(write_file_path_and_name_box_0, "a")
    else:
        box_0_data_txt_file = open(write_file_path_and_name_box_0, "w")
        box_0_data_txt_file.write(output_txt_file_header)


    # ***************************************************
    #  create the required lists and file labels total averages across the replicates (end)
    # ***************************************************

    for job in jobs:

        # *************************
        # drawing in data from single file and extracting specific rows for box 0/zeolite (start)
        # *************************
        reading_file_box_0 = job.fn(output_replicate_txt_file_name_box_0)

        data_box_0 = pd.read_csv(reading_file_box_0, sep='\s+', header=0, na_values='NaN', index_col=False)
        data_box_0 = pd.DataFrame(data_box_0)

        total_molecules_repilcate_box_0 = data_box_0.loc[:, output_column_total_molecules_title][0]
        adsorbed_molecules_repilcate_box_0 = data_box_0.loc[:, output_column_fraction_adsorbed_molecules_title][0]
        Rho_repilcate_box_0 = data_box_0.loc[:, output_column_Rho_title][0]
        adsorbed_Rho_repilcate_box_0 = data_box_0.loc[:, output_column_fraction_adsorbed_Rho_title][0]

        # *************************
        # drawing in data from single file and extracting specific rows for box 0/zeolite (end)
        # *************************

        temp_repilcate_list.append(job.sp.production_temperature_K)
        pressure_repilcate_list.append(job.sp.production_pressure_bar)

        total_molecules_repilcate_box_0_list.append(total_molecules_repilcate_box_0)
        adsorbed_fraction_molecules_box_0_list.append(adsorbed_molecules_repilcate_box_0)
        Rho_repilcate_box_0_list.append(Rho_repilcate_box_0)
        adsorbed_fraction_Rho_box_0_list.append(adsorbed_Rho_repilcate_box_0)

        molecule_name_list.append(job.sp.molecule)
        No_unit_cells_list.append(
            job.doc.No_zeolite_unit_cell_x_axis *
            job.doc.No_zeolite_unit_cell_y_axis *
            job.doc.No_zeolite_unit_cell_z_axis
        )

    # *************************
    # get the replica means and std.devs (start)
    # *************************
    temp_mean = np.mean(temp_repilcate_list)
    temp_std = np.std(temp_repilcate_list, ddof=1)
    pressure_mean = np.mean(pressure_repilcate_list)
    pressure_std = np.std(pressure_repilcate_list, ddof=1)

    total_molecules_mean_box_0 = np.mean(total_molecules_repilcate_box_0_list)
    adsorbed_fraction_molecules_mean_box_0 = np.mean(adsorbed_fraction_molecules_box_0_list)
    Rho_mean_box_0 = np.mean(Rho_repilcate_box_0_list)
    adsorbed_fraction_Rho_mean_box_0 = np.mean(adsorbed_fraction_Rho_box_0_list)

    No_unit_cells_mean_box_0 = np.mean(No_unit_cells_list)

    total_molecules_std_box_0 = np.std(total_molecules_repilcate_box_0_list, ddof=1)
    adsorbed_fraction_molecules_std_box_0 = np.std(adsorbed_fraction_molecules_box_0_list, ddof=1)
    Rho_std_box_0= np.std(Rho_repilcate_box_0_list, ddof=1)
    adsorbed_fraction_Rho_std_box_0 = np.std(adsorbed_fraction_Rho_box_0_list, ddof=1)

    No_unit_cells_std_box_0 = np.std(No_unit_cells_list, ddof=1)

    if len(molecule_name_list) != 0:
        molecule_name = molecule_name_list[0]
        for mol_no_i in range(0, len(molecule_name_list)):
            if molecule_name != molecule_name_list[mol_no_i]:
                raise ValueError("ERROR: There are multiple molecule names in the replicates "
                                 "that are trying to be averaged together. ")

    else:
        raise ValueError("ERROR: There are no molecule names in the job.sp files")

    # *************************
    # get the replica means and std.devs (end)
    # *************************

    # ************************************
    # calcuate the (1) No adsorbed molecules, (2) Rho of adsorbed molecules
    # (3) molecules / unit cell, (4) molecules / volume (start)
    # ************************************

    # calcuate the (1) No adsorbed molecules, (2) Rho of adsorbed molecules
    No_adsorbed_molecules_mean = adsorbed_fraction_molecules_mean_box_0 * total_molecules_mean_box_0
    Rho_of_adsorbed_molecules_mean_kg_per_m_cubed = adsorbed_fraction_Rho_mean_box_0 * Rho_mean_box_0

    No_adsorbed_molecules_std = (
            No_adsorbed_molecules_mean *
            (
                    (total_molecules_std_box_0 / total_molecules_mean_box_0) ** 2 +
                    (adsorbed_fraction_molecules_std_box_0 / adsorbed_fraction_molecules_mean_box_0) ** 2
            ) ** 0.5
    )
    Rho_of_adsorbed_molecules_std_kg_per_m_cubed = (
            Rho_of_adsorbed_molecules_mean_kg_per_m_cubed *
            (
                    (Rho_std_box_0 / Rho_mean_box_0) ** 2 +
                    (adsorbed_fraction_Rho_std_box_0 / adsorbed_fraction_Rho_mean_box_0) ** 2
            ) ** 0.5
    )

    # calcuate the (3) molecules / unit cell
    No_adsorbed_molecules_per_unit_cell_mean = No_adsorbed_molecules_mean / No_unit_cells_mean_box_0

    No_adsorbed_molecules_per_unit_cell_std= (
            No_adsorbed_molecules_per_unit_cell_mean *
            (
                    (No_adsorbed_molecules_std / No_adsorbed_molecules_mean)**2 +
                    (No_unit_cells_std_box_0 / No_unit_cells_mean_box_0)**2
            )**0.5
    )


    # ************************************
    # calcuate the (1) No adsorbed molecules, (2) Rho of adsorbed molecules
    # (3) molecules / unit cell, (4) molecules / volume (end)
    # ************************************

    # ************************************
    # write the analysis data files for box 0/zeolite (start)
    # ************************************

    box_0_data_txt_file.write(
        f"{temp_mean: <30} "
        f"{temp_std: <30} "
        f"{pressure_mean: <30} "
        f"{pressure_std: <30} "
        f"{molecule_name: <30} "
        f"{total_molecules_mean_box_0: <30} "
        f"{total_molecules_std_box_0: <30} "
        f"{adsorbed_fraction_molecules_mean_box_0: <30} "
        f"{adsorbed_fraction_molecules_std_box_0: <30} "
        f"{Rho_mean_box_0: <30} "
        f"{Rho_std_box_0: <30} "
        f"{adsorbed_fraction_Rho_mean_box_0: <30} "
        f"{adsorbed_fraction_Rho_std_box_0: <30} "
        f"{No_adsorbed_molecules_mean: <30} "
        f"{No_adsorbed_molecules_std: <30} "
        f"{Rho_of_adsorbed_molecules_mean_kg_per_m_cubed: <30} "
        f"{Rho_of_adsorbed_molecules_std_kg_per_m_cubed: <30} "
        f"{No_adsorbed_molecules_per_unit_cell_mean: <30} "
        f"{No_adsorbed_molecules_per_unit_cell_std: <30} "
        f" \n"
    )

    # ************************************
    # write the analysis data files for box 0/zeolite (end)
    # ************************************


# ******************************************************
# ******************************************************
# data analysis - get the average and std. dev. from/across all the replicate (end)
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


# ******************************************************
# signac and GOMC-MOSDEF code (end)
# ******************************************************