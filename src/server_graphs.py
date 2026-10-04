import signal
import socket
from subprocess import Popen, TimeoutExpired
import subprocess
import time
from types import FrameType
from typing import override
import paralell


class MyPythonJob(BaseJob):
    def __init__(self, base_path: Path, param: int) -> None:
        super().__init__(base_path)
        self.param = param

    @override
    def name(self) -> str:
        return f"job-param-{self.param}"

    @override
    def get_argv(self) -> list[str]:
        return ["python3", "my_simulation.py", "--param", str(self.param)]