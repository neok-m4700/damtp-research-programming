#!/usr/bin/env python3
import mpi4py
from mpi4py import MPI
import distributed_computing_worker


def main2():
    me = distributed_computing_worker.rankinfo(sizes=[3, 4, 5])
    cartesian_topology = distributed_computing_worker.topology(me)
    ghosts = distributed_computing_worker.ghost_data(cartesian_topology, me.localsizes)
    cartesian_topology.print_info()
    local_array = distributed_computing_worker.initialise_values(me)
    result_l, result_g = distributed_computing_worker.find_max_grad(cartesian_topology, local_array, ghosts)
    distributed_computing_worker.serialised_print(f"Rank {me.rank} had max gradient {result_l} while the global was {result_g}.", cartesian_topology.topology)
    if (me.rank == 0):
        if (distributed_computing_worker.testme(result_g, cartesian_topology)):
            print("Result is correct.")
        else:
            print("Result is incorrect!")
    return result_g, local_array


if (__name__ == "__main__"):
    distributed_computing_worker.main()
