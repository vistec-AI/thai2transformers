#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import multiprocessing



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



"""Blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 2
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        dist.send(tensor=tensor, dst=2)

    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


    """Non-blocking point-to-point communication."""

def run_async(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        req = dist.isend(tensor=tensor, dst=2)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print(f'Rank {rank} started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


if __name__ == '__main__':

    print('multiprocessing.cpu_count()', multiprocessing.cpu_count())

    size = 3
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()