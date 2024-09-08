## signac workflows for MoSDeF-GOMC

### These workflows are from the publication "MoSDeF-GOMC: Python software for the creation of scientific workflows for the Monte Carlo simulation engine GOMC"
--------

### Overview
These signac workflows are designed to automate user workflows for the MoSDeF-GOMC software. The examples contain S8 jet fuel surrogate vapor-liquid equilibrium, neon and radon free energies of hydration in water, and CO2 adsorption in the IRMOF-1.

### IMPORTANT NOTE
In all the project.py files, the user will need to modify the **gomc_binary_path** and/or **namd_binary_path** variables to match the local system's paths for GOMC and NAMD.  The project.py files are located in the following places for each of the projects:
 - [adsorption_CO2_in_IRMOF_1/project/project.py](https://github.com/GOMC-WSU/Publications/tree/main/2023/Crawford_1/adsorption_CO2_in_IRMOF_1/project)
 - [noble_gas_free_energies/project/project.py](https://github.com/GOMC-WSU/Publications/tree/main/2023/Crawford_1/noble_gas_free_energies/project)
 - [S8_vapor_liquid_equilibrium/project/project.py](https://github.com/GOMC-WSU/Publications/tree/main/2023/Crawford_1/S8_vapor_liquid_equilibrium/project)

### Resources
 - [MoSDeF-GOMC Github repository](https://github.com/GOMC-WSU/MoSDeF-GOMC)
 - [MoSDeF-GOMC tutorials and examples](https://github.com/GOMC-WSU/GOMC_Examples/tree/main/MoSDef-GOMC) with [MoSDeF-GOMC YouTube videos](https://www.youtube.com/watch?v=7StVoUCGkHs&list=PLdxD0z6HRx8Y9VhwcODxAHNQBBJDRvxMf)
 - [GOMC Github repository](https://github.com/GOMC-WSU)
 - [Downloading GOMC](https://github.com/GOMC-WSU/GOMC)
 - [Installing GOMC via GOMC manual](https://github.com/GOMC-WSU/Manual)
 - [GOMC YouTube channel](https://www.youtube.com/channel/UCueLGE6tuOyu-mvxIt-U1HQ/playlists)
 - [MoSDeF tools](https://mosdef.org/)
 - [signac](https://signac.io)

### Citation

Please cite this GitHub repository, the MoSDeF-GOMC and signac software.

This repository:  Crawford, B.; Potoff, J. signac workflows for MoSDeF-GOMC. 2022; https://github.com/GOMC-WSU/Publications/tree/main/2022/Crawford_1
 - MoSDeF-GOMC, GOMC, and MoSDeF tools, which are provided [here](https://mosdef-gomc.readthedocs.io/en/latest/reference/citing_mosdef_gomc_python.html)
 - The signac citations are provided [here](https://docs.signac.io/en/latest/acknowledge.html)

 ### Installation

These signac workflows for MoSDeF-GOMC can be built using conda:

`conda env create -f environment.yml`

`conda activate 2022_crawford_1`

 ### Running all the simulations in a given project:
  -  `cd xxxx` (go to the directory with the init.py and project.py files)
  -  `python init.py init` (build all the state points)
  -  `python project.py run ` (run all the jobs on a local computer in which you are the administrator)
  -  `python project.py submit ` (submit all the available jobs to an HPC.  Note: this is currently setup for only Wayne State Grid HPC.  Changes will need to be made to the template/grid.sh file if using a different HPC or if a different conda environment name is used...)


 ### Some core python package versions compatible with this workflow.  
 #### Note: Other versions may not be compatible.
   - MoSDeF-GOMC:       1.0.0
   - signac:            1.7.0
   - signac-dashboard:  0.3.0
   - signac-flow:       0.20.0