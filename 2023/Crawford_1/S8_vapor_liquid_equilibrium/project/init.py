"""Initialize signac statepoints."""

import os
import numpy as np
import signac
import unyt as u

# *******************************************
# the main user varying state points (start)
# *******************************************

project=signac.init_project('S8_jet_fuel_project')

# [400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640] * u.K
production_temperatures = [400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640] * u.K

# [0, 1, 2, 3, 4]
replicas = [0, 1, 2, 3, 4]


# *******************************************
# the main user varying state points (end)
# *******************************************


print("os.getcwd() = " +str(os.getcwd()))

pr_root = os.path.join(os.getcwd(), "src")
pr = signac.get_project(pr_root)

# ignore statepoints that are not being tested (gemc only for methane, pentane)
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
