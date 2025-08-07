"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************
project=signac.get_project()

production_temperatures = [300, 325, 350, 375, 400, 425 ] * u.K

replicas = [0]

# *******************************************
# the main user varying state points (end)
# *******************************************


print("os.getcwd() = " +str(os.getcwd()))

pr_root = os.getcwd()
pr = signac.get_project(pr_root)

# filter the list of dictionaries
total_statepoints = list()

for prod_temp_i in production_temperatures:
    for replica_i in replicas:
        statepoint = {
            "production_temperature_K": np.round(prod_temp_i.to_value("K"), ).item(),
            "replica_number_int": replica_i,
        }
        total_statepoints.append(statepoint)

for sp in total_statepoints:
    pr.open_job(
        statepoint=sp,
    ).init()
