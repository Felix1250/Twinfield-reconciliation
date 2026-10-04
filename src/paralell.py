#!/usr/bin/env python3

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
from typing import override


class BaseJob:
    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = base_path
        self.popen: Popen[bytes] | None = None
        self.start_time: float | None = None

    def name(self) -> str:
        raise NotImplemented

    def msg(self, msg: str):
        print(f"[{self.name()}] {msg}")

    def job_dir(self, mkdir: bool = False) -> Path:
        job_dir = self.base_path / self.name()
        if mkdir:
            job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def exit_code_file(self) -> Path:
        return self.job_dir() / "exitcode.txt"

    def needs_to_run(self) -> bool:
        job_dir = self.job_dir()

        if not job_dir.exists():
            # job dir does not exist, so in a potential previous run we never even ran
            self.msg(f"job dir does not exist => need to run")
            return True

        exit_code_file = self.exit_code_file()
        if not exit_code_file.exists():
            # what ever happened, we don't have an exit code, run again
            self.msg(f"exit code file does not exist => need to run")
            return True

        with open(exit_code_file, "rt") as f:
            line = f.readline()
            try:
                exitcode = int(line)

                # ok, we have a valid exit code
                # do not run again if the previous run exited successfully
                if exitcode == 0:
                    self.msg("previous exitcode was 0, no need to run")
                    return False
                else:
                    self.msg(f"previous exitcode was {exitcode}, need to run")
                    return True
            except ValueError:
                # whatever this file contains is not a valid exit code
                self.msg(f"previous exitcode was invalid '{line}', need to run")
                return True

    def get_argv(self) -> list[str]:
        raise NotImplemented

    def start_process(self):
        job_dir = self.job_dir(mkdir=True)
        argv = self.get_argv()

        self.start_time = time.perf_counter()
        with open(job_dir / "stdout.txt", "wb") as out_fd, open(
            job_dir / "stderr.txt", "wb"
        ) as err_fd:
            p = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out_fd,
                stderr=err_fd,
                # cwd=job_dir,
                start_new_session=True,
            )

        self.msg(f"started with PID {p.pid}")

        self.popen = p

    def terminate(self):
        if not self.popen:
            return
        self.popen.terminate()

    def wait(self, timeout: float | None = None) -> int | None:
        if not self.popen:
            return None

        try:
            rc = self.popen.wait(timeout=timeout)

            if not self.start_time:
                raise RuntimeError("process exited without start_time set")
            stop_time = time.perf_counter()
            wall_time = stop_time - self.start_time

            self.msg(f"exited with {rc} after {wall_time} s")

            with open(self.exit_code_file(), "wt") as f:
                _ = f.write(f"{rc}\n")

            return rc
        except TimeoutExpired:
            return None


class WalkerDeltaJob(BaseJob):
    def __init__(self, base_path: Path, t: int, p: int, f: int) -> None:
        super().__init__(base_path)
        self.t: int = t
        self.p: int = p
        self.f: int = f

    @override
    def name(self) -> str:
        return f"{self.t}-{self.p}-{self.f}"

    @override
    def get_argv(self) -> list[str]:
        java = shutil.which("java")
        if not java:
            raise RuntimeError("could not find java binary")

        ver = "0.1-SNAPSHOT"
        jar = Path(f"sat-link-sim-{ver}.jar")

        # fallback for running from the source root
        if not jar.exists():
            maven_target = Path("target")
            if maven_target.is_dir():
                jar = maven_target / f"sat-link-sim-{ver}-jar-bin-dist" / jar

        if not jar.exists():
            raise RuntimeError(f"could not find {jar}")

        return [
            java,
            f"-Dorekit.data.path={ Path.home() / "orekit-data" }",
            "-jar",
            str(jar),
            #"--timespan=-2",
            "--step=0.01",
            f"-t={self.t}",
            f"-p={self.p}",
            f"-f={self.f}",
        ]


def generate_jobs(base_path: Path) -> Generator[BaseJob]:
    for t, p in [(900, 30), (400, 20), (100, 10)]:
        for f in range(0, p):
            yield WalkerDeltaJob(base_path, t, p, f)


def schedule(gen: Generator[BaseJob], max_parallel_jobs: int | None = None):
    if max_parallel_jobs is None:
        max_parallel_jobs = len(os.sched_getaffinity(0))
    print(f"running {max_parallel_jobs} jobs in parallel")

    # set up signal handler for graceful shutdown
    # this is the selfpipe trick https://cr.yp.to/docs/selfpipe.html
    # not sure this required in Python or if a simple flag would also do the job
    # but it's what the docs recommend and we can block on it
    # https://docs.python.org/3/library/signal.html#note-on-signal-handlers-and-exceptions
    interrupt_read, interrupt_write = socket.socketpair()
    interrupt_read.setblocking(False)

    def signal_handler(
        signum: int, frame: FrameType | None  # pyright: ignore[reportUnusedParameter]
    ):
        try:
            b = signum.to_bytes(1)
        except OverflowError:
            b = b"\0"
        _ = interrupt_write.send(b)

    # handle Ctrl+C and SIGTERM by killing all running jobs and exiting
    _ = signal.signal(signal.SIGINT, signal_handler)
    _ = signal.signal(signal.SIGTERM, signal_handler)

    running: list[BaseJob] = []
    should_exit = False
    while True:
        # filter out completed jobs, so keep only those where the timeout expired
        running = list(filter(lambda j: j.wait(timeout=0) is None, running))

        while not should_exit and len(running) < max_parallel_jobs:
            try:
                new_job = next(gen)
                if new_job.needs_to_run():
                    new_job.start_process()
                    running.append(new_job)
            except StopIteration:
                # generator does not have any further jobs,
                # so we have to break from the while as the condition can
                # now not be fulfilled anymore
                break

        if len(running) == 0:
            # nothing running anymore and the attempt to fill the queue just now
            # also did nothing, so we're done
            break

        # check if we got a signal and should therefore shutdown
        # use this opportunity to wait a bit before the next loop iteration
        timeout = 0.5 if should_exit else 5
        ready = select.select([interrupt_read], [], [], timeout)
        if len(ready[0]) > 0:
            # our signal handlers wrote a byte, time to exit
            sig = interrupt_read.recv(1)
            print(
                f"received signal {int.from_bytes(sig)}, killing running jobs and exiting"
            )
            should_exit = True

            # terminate all jobs
            # we assume they are well-behaved and SIGTERM is enough for them to exit prematurely
            for j in running:
                j.terminate()


def main():
    result_dir: Path = Path.cwd() / "results"
    gen = generate_jobs(result_dir)
    schedule(gen)


if __name__ == "__main__":
    main()