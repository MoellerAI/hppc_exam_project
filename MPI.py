import numpy as np 
from mpi4py import MPI
import mpi4py
import time
#mpi4py.rc.initialize = False
#mpi4py.rc.finalize = False
n_cuts = 3
n_settings = n_cuts**8
NO_MORE_TASKS = n_settings+1

class Data():
    """Process and load data
    """
    def __init__(self, filename: str = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv') -> None:
        """setting initial data specific parameters.

        Args:
            filename (str, optional): data file to load. Defaults to 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'.
        """
        self.data = self.read_data(filename)
        self.nevents = self.data.shape[0]
        self.name = ["averageInteractionsPerCrossing", "p_Rhad","p_Rhad1", "p_TRTTrackOccupancy", "p_topoetcone40", 
                     "p_eTileGap3Cluster", "p_phiModCalo", "p_etaModCalo"]
        self.NvtxReco = self.data[:,1]
        self.p_nTracks = self.data[:,2]
        self.p_truthType = self.data[:,10]

        self.signal = self.p_truthType == 2

        self.data = self.data[:, [0,3,4,5,6,7,8,9]]

        self.means_sig = np.array([np.average(self.data[self.signal, i]) for i in range(8)])
        self.means_bckg = np.array([np.average(self.data[~self.signal, i]) for i in range(8)])
        self.flip = np.sign(self.means_bckg - self.means_sig)
            
        for i in range(8): 
            self.data[:, i] *= self.flip[i]
            self.means_sig[i]  = self.means_sig[i] * self.flip[i]
            self.means_bckg[i] = self.means_bckg[i] * self.flip[i]


    
    def read_data(self, filename = 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv') -> np.array:
        """Read data file using numpy (fastest).

        Args:
            filename (str, optional):File to load. Defaults to 'mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv'.

        Returns:
            np.array: Returns data file except counter column and header row.
        """
        return np.loadtxt(filename, delimiter = ',', skiprows = 1, usecols=range(1,12))

def master(nworker: int, ds: Data, comm):
    """Master function code. Master needs to collect work from workers.

    Args:
        nworker (int): The number of workers.
        ds (Data): Dataset as a Data class object.
    """
    print('im master')
    ranges = np.zeros([n_cuts, 8])
    settings = list()
    accuracy = np.zeros(n_settings)

    for j in range(n_cuts):
        ranges[j] = ds.means_sig + j * (ds.means_bckg - ds.means_sig) / n_cuts

    for k in range(n_settings):
        div = 1
        _set = np.zeros(8)
        for i in range(8):
            idx = int((k/div) % n_cuts)
            _set[i] = ranges[idx][i]
            div *= n_cuts
        settings.append(_set)
    
    tstart = time.time()

    for worker in range(1, nworker+1):
        req = comm.isend(settings[worker-1][0], worker, worker-1)
        req.Free()

    task_send = nworker
    status = MPI.Status()
    while task_send < n_settings:
        acc = comm.recv(status = status)
        accuracy[status.Get_tag()] = acc
        req = comm.isend(settings[task_send], status.Get_source(), task_send)
        req.Free()
        task_send = task_send + 1

    for worker in range(1, nworker+1):
        acc = comm.recv(status = status)
        accuracy[status.Get_tag()] = acc
        req = comm.isend(settings[0], status.Get_source(), NO_MORE_TASKS)
        req.Free()


    # for k in range(n_settings):
    #     accuracy.append(task_function(settings[k], ds))
    
    tend = time.time()

    print(len(accuracy))

    idx_best = np.argmax(accuracy)
    best_accuracy_score = accuracy[idx_best]
    
    print("Best accuracy obtained:", best_accuracy_score, "\n")
    print("Final cuts: \n")
    
    for i in range(8):
        print(ds.name[i], " : ", settings[idx_best][i]*ds.flip[i], "\n")
    
    print()
    print("Number of settings:", n_settings, "\n")
    print("Elapsed time:", (tend - tstart), "\n")
    print("task time [mus]:", (tend - tstart)/ n_settings, "\n")


def worker(rank: int, ds: Data, comm):
    print('im a worker')
    task = np.zeros(8)
    status = MPI.Status()
    task = comm.recv(source = 0, status = status)
    while status.Get_tag() < NO_MORE_TASKS:
        acc = task_function(task, ds)
        req = comm.isend(acc, 0, status.Get_tag())
        req.Free()
        task = comm.recv(source = 0, status = status)

    
def task_function(setting: np.array, ds: Data) -> float:
    """_summary_

    Args:
        setting (numpy array): _description_
        ds (Data): _description_

    Returns:
        float: _description_
    """
    pred = np.min(ds.data < setting, axis=1)
    return np.sum(pred == ds.signal) / ds.nevents

def main() -> None:

    #MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nrank = comm.Get_size()
    print(comm)
    

    ds = Data()

    if rank == 0:
        master(nrank-1, ds, comm)
    else:
        worker(rank, ds, comm)
    
    #MPI.Finalize()

if __name__== "__main__" :
    main()