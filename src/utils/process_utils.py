import subprocess


def get_pid_by_grep(cmd_str):
    # Run the ps command and grep for the process
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
    # Decode the output and split into lines
    processes = result.stdout.decode().splitlines()
    # Filter lines that contain the command string
    for process in processes:
        if cmd_str in process:
            # Split the line into parts and return the PID (second column)
            return int(process.split()[1])
    return None