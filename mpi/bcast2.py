import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # create a data array on process 0
    # in real code, this section might
    # read in data parameters from a file
    numData = 5
    data = np.linspace(0.0, 3.14, numData)
    comm.bcast(numData, root=0)
else:
    numData = comm.bcast(None, root=0)
    data = np.empty(numData, dtype="d")

comm.Bcast(data, root=0)  # broadcast the array from rank 0 to all others


print("Rank: ", rank, ", data received: ", data)
