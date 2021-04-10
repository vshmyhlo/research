import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # new: gives number of ranks in comm
rank = comm.Get_rank()

chunk_size = 5
data = None
if rank == 0:
    data = np.arange(0, chunk_size * size).astype(np.float32)
    # when size=4 (using -n 4), data = [1.0:40.0]


recvbuf = np.empty(chunk_size, dtype=np.float32)  # allocate space for recvbuf
comm.Scatter(data, recvbuf, root=0)


print("Rank: ", rank, ", recvbuf received: ", recvbuf)
