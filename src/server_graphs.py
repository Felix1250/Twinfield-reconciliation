#!/usr/bin/env python3
from __future__ import annotations

"""
run single-threaded simulations ("jobs") in parallel, one job per available CPU

The simulations to run are defined in generate_jobs(),
this generator yields (subclasses of) BaseJob which are wrappers that
start the simulations to run as child processes.
You probably want to create an own subclass of BaseJob for your particular
simulation to customize the job name and the process to start.
The only methods you should need to override are the constructor (duh!),
name (defines the output directory of the job) and get_argv (what to execute).

Each job is allocated one directory in the result_dir (see main function)
named after it's name().
This directory also contains the stdout, stderr and the exitcode of the run.

If invoked with an existing result_dir, jobs that already ran successfully
(recorded exitcode == 0) are skipped.
"""

from collections.abc import Generator
import os
from pathlib import Path
import select
import shutil
import signal
import socket
from subprocess import Popen, TimeoutExpired
import subprocess
import time
from types import FrameType
import numpy as np


try:
    from typing import override
except ImportError:
    def override(func):
        return func

import paralell
import general_graphs
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Twinfield_job(paralell.BaseJob):
    def __init__(
        self,
        base_path: Path,
        savefile: str,
        pd: float = math.pow(10, -5),
        eta: float = 0.8,
        ea: float = 0.15,
        distance: float = 0,
        pz = 1 - 0.269,
        mu = 0.422,
        length = 10000000
    ) -> None:
        super().__init__(base_path)
        self.savefile = savefile
        self.pd = pd
        self.eta = eta
        self.ea = ea
        self.distance = distance
        self.pz = pz
        self.mu = mu
        self.length = length


    @override
    def name(self) -> str:
        return f"twinfield_pd1{self.pd}_eta{self.eta}_ea{self.ea}_dist{self.distance}_pz{self.pz}_mu{self.mu}_length{self.length}"

    def get_argv(self) -> list[str]:
        return [
            "python3",
            "src/general_graphs.py",  
            "--savefile",
            str(self.savefile),
            "--pd",
            str(self.pd),
            "--eta",
            str(self.eta),
            "--ea",
            str(self.ea),
            "--distance",
            str(self.distance),
            "--pz",
            str(self.pz),
            "--mu",
            str(self.mu),
            "--length",
            str(self.length),
        ]

'''
---------------------------------------------------------------------------------
what kind of jobs to run, one job per CPU core, each job is a single-threaded simulation
---------------------------------------------------------------------------------
'''

datapoints = 20

def generate_pd_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 10
    savefolder = os.getcwd() + "/saves/pd3"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        yield Twinfield_job(
            base_path=base_path,
            savefile=savefile,
            pd=math.pow(10, -i),
            eta=0.8,
            ea=0.15,
            distance=0,
            length = 100000000
        )

def generate_dist_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 30
    savefolder = os.getcwd() + "/saves/dist2"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        savefile = savefolder + "/_dist" + str(i) + ".npz"
        yield Twinfield_job(
            base_path=base_path,
            savefile=savefile,
            pd=math.pow(10, -5),
            eta=0.8,
            ea=0.15,
            distance=i/datapoints*350, #max 350km one side so 700km total
            length = 100000000
        )


def generate_eta_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    savefolder = os.getcwd() + "/saves/eta"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        savefile = savefolder + "/_eta" + str(i) + ".npz"
        yield Twinfield_job(
            base_path=base_path,
            savefile=savefile,
            pd=math.pow(10, -5),
            eta=i/20,
            ea=0.15,
            distance=0,
        )


def generate_ea_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    savefolder = os.getcwd() + "/saves/ea"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        savefile = savefolder + "/_ea" + str(i) + ".npz"
        yield Twinfield_job(
            base_path=base_path,
            savefile=savefile,
            pd=math.pow(10, -5),
            eta=0.8,
            ea=i/20,
            distance=0,
        )



def generate_pzmu_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 5
    savefolder = os.getcwd() + "/saves/pzmu"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(7):
        for j in range(5,7):
            savefile = savefolder + "/_pz" + str(i) + "mu" + str(j) +".npz"
            yield Twinfield_job(
                base_path=base_path,
                savefile=savefile,
                mu=0.25 + j/20,
                pz=1- (0.1+i/20)
            )
    
'''
---------------------------------------------------------------------------------
functions to create the jobs and run them in parallel, one job per CPU core
---------------------------------------------------------------------------------
'''

def pd():
    result_dir: Path = Path.cwd() / "results/pd3"
    gen = generate_pd_jobs(result_dir)
    paralell.schedule(gen,max_parallel_jobs=10)

def eta():
    result_dir: Path = Path.cwd() / "results/eta"
    gen = generate_eta_jobs(result_dir)
    paralell.schedule(gen)

def ea():
    result_dir: Path = Path.cwd() / "results/ea"
    gen = generate_ea_jobs(result_dir)
    paralell.schedule(gen)

def dist():
    result_dir: Path = Path.cwd() / "results"
    gen = generate_dist_jobs(result_dir)
    paralell.schedule(gen,max_parallel_jobs=25)

def pzmu():
    result_dir: Path = Path.cwd() / "pzmu"
    gen = generate_pzmu_jobs(result_dir)
    paralell.schedule(gen)

'''
---------------------------------------------------------------------------------
All the aftercomm stuff + either making graphs out of it or just saving it in csvs
---------------------------------------------------------------------------------
'''

def pd_rest():
    savefolder = os.getcwd() + "/saves/pd2"
    datapoints = 6
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)
    labels = []

    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        labels.append(f"$10^{{-{i}}}$")
        signal_length[i], error_rate[i] , key_rate[i] , comm_iters[i]=general_graphs.simulate_comm(savefile,aopp_enabled = True,ldpc_enabled = True, hash_enabled=True,enable_twinfield=False)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/pd2_"  + ".csv")

def pd_rest_aopp():
    savefolder = os.getcwd() + "/saves/pd"
    datapoints =6
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        labels.append(f"$10^{{-{i}}}$")
        signal_length[i], error_rate[i] , key_rate[i] , comm_iters[i]=general_graphs.simulate_comm(savefile,aopp_enabled=True)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/pd2_"  + ".csv")
        
def eta_rest():
    savefolder = os.getcwd() + "/saves/eta"
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_eta" + str(i) + ".npz"
        labels.append(f"{i/20:.2f}")
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/figures/my_graph/eta.png",x_label="'eta",csv_savepath = os.getcwd() + "/figures/my_graph/eta.csv")

def ea_rest():
    savefolder = os.getcwd() + "/saves/pd2"
    datapoints = 20
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)
    ephi = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(5) + ".npz"
        ea = i/ (datapoints ) * (0.06)
        labels.append(f"{ea:.2f}")
        signal_length[i], error_rate[i], key_rate[i], comm_iters[i],ephi[i]=general_graphs.simulate_comm_ea(savefile,ea,aopp_enabled = True,ldpc_enabled = True, enable_twinfield=True,hash_enabled=True)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,ephi=ephi,csv_savepath = os.getcwd() + "/figures/my_graph/ea_phi3.csv")

def lambda_rest():
    savefolder = os.getcwd() + "/saves/pd2"
    datapoints = 10
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)
    ephi = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(5) + ".npz"
        ea = 0.012
        lam = i/100*5 + 0.05
        labels.append(f"{lam:.2f}")
        signal_length[i], error_rate[i], key_rate[i], comm_iters[i],ephi[i]=general_graphs.simulate_comm_ea(savefile,ea,aopp_enabled = True,ldpc_enabled = True, enable_twinfield=True,hash_enabled=True,lam = lam)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,ephi=ephi,csv_savepath = os.getcwd() + "/figures/my_graph/lambda2.csv")

def ldpc_rest(discl_k):
    savefolder = os.getcwd() + "/saves/pd2"
    labels = []
    savefile = savefolder + "/_pd" + str(5) + ".npz"
    ea = 0
    signal_length, error_rate, key_rate, comm_iters,f,labels =general_graphs.simulate_comm_ldpc(savefile,ea,aopp_enabled = True,ldpc_enabled = True,hash_enabled=True,discl_k=discl_k)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,fres = f,csv_savepath = os.getcwd() + "/figures/my_graph/ldpc2_" + str(discl_k)+".csv")

def dist_ldpc(enable_twinfield=True,aopp_enabled = False,hash_enabled = True,counter = 0):
    savefolder = os.getcwd() + "/saves/dist"
    datapoints = 44
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_dist" + str(i) + ".npz"
        labels.append(i/50*700)
        signal_length[i], error_rate[i], key_rate[i], comm_iters[i]=general_graphs.simulate_comm(savefile,aopp_enabled = aopp_enabled,ldpc_enabled = True, enable_twinfield=enable_twinfield,hash_enabled=hash_enabled)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/dist_" + str(enable_twinfield) +"_" + str(counter)+ ".csv")



def dist_general(things = 0,aopp_enabled = True,hash_enabled = True,ldpc_enabled = True):
    savefolder = os.getcwd() + "/saves/dist"
    datapoints = 50
    comm_iters = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    signal_length = np.zeros(datapoints)
    error_rate = np.zeros(datapoints)   
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_dist" + str(i) + ".npz"
        labels.append(i/50*700)
        signal_length[i], error_rate[i], key_rate[i], comm_iters[i]=general_graphs.simulate_comm(savefile,aopp_enabled = aopp_enabled,ldpc_enabled = ldpc_enabled,hash_enabled = hash_enabled, enable_twinfield=True)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/dist1" + str(things) + ".csv")

def pzmu_rest(aopp_enabled = True,hash_enabled = True,ldpc_enabled = True):
    things = 0
    savefolder = os.getcwd() + "/saves/pzmu"
    datapoints = 5
    for i in range(7):
        comm_iters = np.zeros(datapoints)
        key_rate = np.zeros(datapoints)
        signal_length = np.zeros(datapoints)
        error_rate = np.zeros(datapoints)
        labels = []
        for j in range(datapoints):
            savefile =savefolder + "/_pz" + str(i) + "mu" + str(j) +".npz"
            labels.append(i*100+j)
            signal_length[j], error_rate[j], key_rate[j], comm_iters[j]=general_graphs.simulate_comm(savefile,aopp_enabled = aopp_enabled,ldpc_enabled = ldpc_enabled,hash_enabled = hash_enabled, enable_twinfield=True)
        things += 1
        general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/pzmu" + str(things) + ".csv")



if __name__ == "__main__":
    #ea()
    #eta()
    #pd()
    #eta_rest()
    #ldpc_rest(0.5)
    #ldpc_rest(0.75)
    #ldpc_rest(1)
    #lambda_rest()
    #ea_rest()
    #pd_rest()
    #pd_rest_aopp()
    #dist_ldpc(enable_twinfield=True,aopp_enabled = False,hash_enabled = True,counter = 0)
    #dist_ldpc(enable_twinfield=False,aopp_enabled = True,hash_enabled = True,counter = 1)
    #dist()
    #pd()
    #pzmu()
    pzmu_rest()

    #
    #dist_ldpc(enable_twinfield = True)
    #dist_ldpc(enable_twinfield = False)
    #pzmu_rest()
