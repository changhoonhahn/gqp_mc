import time
import random
import dask_jobqueue
import dask 
from dask.distributed import Client 

def costly_simulation(list_param):
    time.sleep(random.random())
    return list_param


cluster = dask_jobqueue.SLURMCluster(cores=1, processes=2, memory='16GB')
#cluster.scale(1)
client = Client(cluster)

thetas = range(10) 

for _theta in thetas: 
    print(costly_simulation(_theta))


lazys = [] 
for _theta in thetas: 
    lazy = dask.delayed(costly_simulation)(_theta)
    lazys.append(lazy) 
results = dask.compute(*lazys) 

print(results) 
