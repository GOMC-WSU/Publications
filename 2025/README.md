## signac workflows for MoSDeF-GOMC

### These workflows are from the publication "Extension of Molecular Exchange Monte Carlo for the Prediction of Liquid-Liquid Equilibria in Gibbs Ensemble Monte Carlo Simulations"
--------

### Resources
 - [MoSDeF-GOMC Github repository](https://github.com/GOMC-WSU/MoSDeF-GOMC)
 - [MoSDeF-GOMC tutorials and examples](https://github.com/GOMC-WSU/GOMC_Examples/tree/main/MoSDef-GOMC) with [MoSDeF-GOMC YouTube videos](https://www.youtube.com/watch?v=7StVoUCGkHs&list=PLdxD0z6HRx8Y9VhwcODxAHNQBBJDRvxMf)
 - [GOMC Github repository](https://github.com/GOMC-WSU)
 - [Downloading GOMC](https://github.com/GOMC-WSU/GOMC)
 - [Installing GOMC via GOMC manual](https://github.com/GOMC-WSU/Manual)
 - [GOMC YouTube channel](https://www.youtube.com/channel/UCueLGE6tuOyu-mvxIt-U1HQ/playlists)
 - [MoSDeF tools](https://mosdef.org/)
 - [signac](https://signac.io)

### Installation

These signac workflows for MoSDeF-GOMC can be built using conda:

`conda env create -f environment.yml`

`conda activate mg3`

You will need to replace gmso_charmm_writer.py in  ~/conda/envs/mg3/lib/python3.10/site-packages/mosdef_gomc/formats with gmso_charmm_writer.py in this repository to 
run the perfluorobutane-butane and perfluorohexane-hexane examples.  This file has been modified to support 7 term periodic dihedral potentials.

 ### To run specific simulations:
  -  `cd xxxx` (go to the directory with the init.py and GEMC.py (or GEMC-LLE.py or GEMC-NPT.py) files)
  -  `signac init` (initialize workspace)
  -  `python init.py ` (build all the state points)
  -  `python GEMC.py run ` (run calculations directly)
  -  `python GEMC.py submit ` (Submit to a job scheduler.  The template file is designed to work with our cluster, which uses Slurm to manage jobs).
