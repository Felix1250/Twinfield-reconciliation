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
    ) -> None:
        super().__init__(base_path)
        self.savefile = savefile
        self.pd = pd
        self.eta = eta
        self.ea = ea
        self.distance = distance


    @override
    def name(self) -> str:
        return f"twinfield_pd5{self.pd}_eta{self.eta}_ea{self.ea}_dist{self.distance}"

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
        ]


datapoints = 20
savefolder = os.getcwd() + "/saves/pd"
def generate_jobs(base_path: Path) -> Generator[paralell.BaseJob]:
    # Example parameter sweeps; customize these tuples as needed for your server graphs
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
        )
    


def pd():
    print(savefolder)
    result_dir: Path = Path.cwd() / "results"
    gen = generate_jobs(result_dir)
    paralell.schedule(gen)

def pd_rest():
    error_rate = np.zeros(datapoints)
    key_rate = np.zeros(datapoints)
    labels = []
    for i in range(datapoints):
        savefile = savefolder + "/_pd" + str(i) + ".npz"
        labels.append(i)
        signal_length, error_rate[i] , key_rate[i] , comm_iters=general_graphs.simulate_comm(savefile)
    general_graphs.plot_general(key_rate, error_rate,labels,os.getcwd() + "/myfigures/savepath.png")
        



if __name__ == "__main__":
    pd()
    #pd_rest()