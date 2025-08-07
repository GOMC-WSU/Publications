"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************
project=signac.get_project()
production_temperatures = [373.26]*u.K
production_pressures = [50,100,172,250,340,400,444,500]*u.bar
replicas = [0,1,2]

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
        for replica_i in replicas:
            statepoint = {
                    "production_temperature_K": prod_temp_i.to_value("K"), "production_pressure_bar":prod_press_i.to_value("bar"),
                "replica_number_int": replica_i,
            }
            total_statepoints.append(statepoint)

for sp in total_statepoints:
    pr.open_job(
        statepoint=sp,
    ).init()
