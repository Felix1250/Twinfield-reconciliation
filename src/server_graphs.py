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
    savefolder = os.getcwd() + "/saves/pd"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        savefile = savefolder + "/_pd2" + str(i) + ".npz"
        yield Twinfield_job(
            base_path=base_path,
            savefile=savefile,
            pd=math.pow(10, -i*10/20),
            eta=0.8,
            ea=0.15,
            distance=0,
            length = 1000000000
        )

def generate_dist_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 50
    savefolder = os.getcwd() + "/saves/dist"
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

def generate_dist_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 50
    savefolder = os.getcwd() + "/saves/dist"
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
        )

def generate_pzmu_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    datapoints = 5
    savefolder = os.getcwd() + "/saves/pzmu"
    Path(savefolder).parent.mkdir(parents=True, exist_ok=True)
    for i in range(datapoints):
        for j in range(datapoints):
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
    result_dir: Path = Path.cwd() / "results/pd2"
    gen = generate_pd_jobs(result_dir)
    paralell.schedule(gen,max_parallel_jobs=6)

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
    paralell.schedule(gen)

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
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    datapoints = 10
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        labels.append(f"$10^{{-{i}}}$")
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/figures/my_graph/pd.png",x_label="'$p_d$",csv_savepath = os.getcwd() + "/figures/my_graph/pd.csv")

def pd_rest_aopp():
    savefolder = os.getcwd() + "/saves/pd"
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        labels.append(f"$10^{{-{i}}}$")
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile,aopp_enabled=True)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/figures/my_graph/pd_aopp.png",x_label="'$p_d$",csv_savepath = os.getcwd() + "/figures/my_graph/pd_aopp.csv")
        
def eta_rest():
    savefolder = os.getcwd() + "/saves/eta"
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_eta" + str(i) + ".npz"
        labels.append(f"{i/20:.2f}")
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/figures/my_graph/eta.png",x_label="'$\eta$",csv_savepath = os.getcwd() + "/figures/my_graph/eta.csv")

def ea_rest():
    savefolder = os.getcwd() + "/saves/ea"
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_ea" + str(i) + ".npz"
        labels.append(f"{i/20:.2f}")
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/figures/my_graph/ea.png",x_label="'$e_a$",csv_savepath = os.getcwd() + "/figures/my_graph/ea.csv")

def dist_ldpc(enable_twinfield=True,aopp_enabled = False,hash_enabled = True):
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
        signal_length[i], error_rate[i], key_rate[i], comm_iters[i]=general_graphs.simulate_comm(savefile,aopp_enabled = aopp_enabled,ldpc_enabled = True, enable_twinfield=enable_twinfield,hash_enabled=hash_enabled)
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/dist_" + str(enable_twinfield) + ".csv")

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
    general_graphs.save_as_csv(signal_length, error_rate , key_rate , comm_iters,labels,csv_savepath = os.getcwd() + "/figures/my_graph/dist" + str(things) + ".csv")

def pzmu_rest(aopp_enabled = True,hash_enabled = True,ldpc_enabled = True):
    things = 0
    savefolder = os.getcwd() + "/saves/pzmu"
    datapoints = 5
    for i in range(datapoints):
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
    pd()
    #pd_rest()
    #dist()
    #pzmu()

    #pd_rest_aopp()
    #dist_ldpc(enable_twinfield = True)
    #dist_ldpc(enable_twinfield = False)
    #pzmu_rest()
