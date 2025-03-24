import os
import subprocess


def get_pid_by_grep(cmd_str):
    # Get our own process ID to filter it out
    own_pid = os.getpid()

    # Run the ps command
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)

    # Decode the output and split into lines
    processes = result.stdout.decode().splitlines()

    # Filter lines that contain the command string, but exclude our own process
    for process in processes:
        parts = process.split()
        if len(parts) > 1:
            pid = int(parts[1])
            # Skip our own process and check for the command string
            if pid != own_pid and cmd_str in process:
                return pid
