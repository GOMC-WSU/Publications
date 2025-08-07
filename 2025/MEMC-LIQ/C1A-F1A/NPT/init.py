"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************

#project=signac.init_project()
project=signac.get_project()
#production_temperatures = [275, 300, 325, 350, 375, 400, 425, 450, 475 ] * u.K
production_temperatures = [90  ] * u.K
production_compositions = [0.233,0.763]
#production_temperatures = [86,87,88,89,90,91,92]*u.K
#production_temperatures = [88]*u.K
#production_pressures = [11.112] *u.bar
production_pressures = [1.0]*u.bar
#production_pressures = [0.65 ]*u.bar

# [0, 1, 2, 3, 4]
replicas = [0]

# *******************************************
# the main user varying state points (end)
# *******************************************


print("os.getcwd() = " +str(os.getcwd()))

pr_root = os.getcwd()
pr = signac.get_project(pr_root)

# filter the list of dictionaries
total_statepoints = list()
for prod_press_i in production_pressures:
    for prod_temp_i in production_temperatures:
        for prod_comp_i in production_compositions:
            for replica_i in replicas:
                statepoint = {
                    "production_temperature_K": prod_temp_i.to_value("K"), "production_pressure_bar":prod_press_i.to_value("bar"),
                "replica_number_int": replica_i, "production_composition":prod_comp_i
            }
            total_statepoints.append(statepoint)

for sp in total_statepoints:
    pr.open_job(
        statepoint=sp,
    ).init()
