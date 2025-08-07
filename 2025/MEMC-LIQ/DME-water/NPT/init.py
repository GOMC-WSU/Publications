"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************
project=signac.get_project()
production_temperatures = [373.26  ] * u.K
production_compositions = [0.232,0.723]
production_pressures = [250.0]*u.bar

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
