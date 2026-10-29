"""Bounded subprocess execution without changing the environment's cgroup policy."""
import os
from pathlib import Path
import signal
import subprocess
import time


class ResourceLimitError(RuntimeError):
    pass


def run_bounded(command,stdout,stderr,env,cwd,timeout,max_rss_mib):
    if max_rss_mib<=0: raise ValueError('max_rss_mib must be positive')
    started=time.monotonic();peak=0.
    process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=cwd,start_new_session=True)
    try:
        while process.poll() is None:
            try:
                status=Path(f'/proc/{process.pid}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmRSS:'):
                        peak=max(peak,int(line.split()[1])/1024)
            except (FileNotFoundError,ProcessLookupError): pass
            if peak>max_rss_mib:
                raise ResourceLimitError(f'worker exceeded RSS budget {max_rss_mib} MiB (observed {peak:.1f} MiB)')
            if time.monotonic()-started>timeout:
                raise subprocess.TimeoutExpired(command,timeout)
            time.sleep(.1)
        result=subprocess.CompletedProcess(command,process.returncode)
        result.supervisor_peak_rss_mib=peak
        return result
    finally:
        if process.poll() is None:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
