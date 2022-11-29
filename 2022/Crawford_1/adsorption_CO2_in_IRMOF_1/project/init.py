"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************

project=signac.init_project('CO2_IRMOF_1_zeolite_ads')

# ["CO2"]
molecule = ["CO2"]

production_temperatures = [298]* u.K


#production_pressure_to_fugacity_dict: pressure --> bar, fugacity ---> bar

production_pressure_to_fugacity_dict_bar = {
            0.05 : 0.049987386602 ,
            0.1  : 0.099949549987 ,
            0.2  : 0.199798228581 ,
            0.3  : 0.299546078702 ,
            0.4  : 0.399193143224 ,
            0.5  : 0.498739464978 ,
            0.6  : 0.598185086754 ,
            0.7  : 0.697530051294 ,
            0.8  : 0.796774401300 ,
            0.9  : 0.895918179428 ,
            1.0  : 0.994961428290 ,
}


# [0, 1, 2, 3, 4]
replicas = [0, 1, 2, 3, 4]


# *******************************************
# the main user varying state points (end)
# *******************************************


# get the fugacity from the set pressure in bar
production_pressure = []
production_fugacity = []
for pressure_k, fugacity_k in production_pressure_to_fugacity_dict_bar.items():
    production_pressure.append(pressure_k)
    production_fugacity.append(fugacity_k)

print("os.getcwd() = " +str(os.getcwd()))

pr_root = os.path.join(os.getcwd(), "src")
pr = signac.get_project(pr_root)

# ignore statepoints that are not being tested (gemc only for methane, pentane)
# filter the list of dictionaries
total_statepoints = list()

for molecule_i in molecule:
    for prod_temp_i in production_temperatures:
        for prod_pressure_i in range(0, len(production_pressure)):
            for replica_i in replicas:
                statepoint = {
                    "molecule": molecule_i,
                    "production_temperature_K": np.round(prod_temp_i.to_value("K"), decimals=6 ).item(),
                    "production_pressure_bar": np.round(production_pressure[prod_pressure_i], decimals=6).item(),
                    "production_fugacity_bar": np.round(production_fugacity[prod_pressure_i], decimals=16).item(),
                    "replica_number_int": replica_i,
                }

                total_statepoints.append(statepoint)

for sp in total_statepoints:
    pr.open_job(
        statepoint=sp,
    ).init()
