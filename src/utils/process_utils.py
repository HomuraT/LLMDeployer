import os
import subprocess
import logging
import psutil
import signal
import time

logging.basicConfig(level=logging.INFO)

def get_pid_by_grep(pattern: str) -> list[int]:
    """
    Find process IDs by grepping the command line.

    Args:
        pattern (str): The pattern to search for in the command line.

    Returns:
        list[int]: A list of matching process IDs.
    """
    pids = []
    try:
        for proc in psutil.process_iter(['pid', 'cmdline']):
            if proc.info['cmdline'] and pattern in " ".join(proc.info['cmdline']):
                pids.append(proc.info['pid'])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    except Exception as e:
        logging.error(f"Error finding PID by grep pattern '{pattern}': {e}")
    return pids

def _terminate_process_tree(pid: int, proc: psutil.Process = None):
    """
    Attempts to terminate a process and its children gracefully (TERM) then forcefully (KILL).
    Args:
        pid (int): The process ID to terminate.
        proc (psutil.Process, optional): Existing psutil.Process object if available.
    """
    try:
        if proc is None:
            proc = psutil.Process(pid)
        
        # Get children first
        children = []
        try:
            children = proc.children(recursive=True)
        except psutil.Error as e:
            logging.warning(f"Could not get children for PID {pid}, continuing termination: {e}")

        # Terminate children
        if children:
            logging.info(f"Terminating child processes of {pid}...")
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass # Already gone
            gone, alive = psutil.wait_procs(children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

        # Terminate parent
        logging.info(f"Attempting to terminate process {pid}...")
        try:
            proc.terminate() # SIGTERM
            proc.wait(timeout=3)
            logging.info(f"Terminated process {pid} with SIGTERM.")
            return True # Indicate successful termination
        except psutil.TimeoutExpired:
            logging.warning(f"Process {pid} did not terminate after SIGTERM. Sending SIGKILL...")
            proc.kill() # SIGKILL
            proc.wait()
            logging.info(f"Terminated process {pid} with SIGKILL.")
            return True # Indicate successful termination (forceful)
        except psutil.NoSuchProcess:
            logging.info(f"Process {pid} already exited before SIGTERM.")
            return True # Already gone

    except psutil.NoSuchProcess:
        logging.info(f"Process {pid} not found for termination.")
        return True # Consider not found as 'cleaned'
    except Exception as e:
        logging.error(f"Error during termination of process {pid} tree: {e}")
        return False # Indicate failure

def cleanup_potential_vllm_orphans():
    """
    Scans for and attempts to terminate potential orphaned processes 
    related to VLLM, including the main server process and its multiprocessing helpers.
    """
    logging.info("Scanning for potential orphaned VLLM-related processes at startup...")
    cleaned_pids = set()
    processes_to_check = []

    # --- First pass: Collect potential VLLM related processes --- 
    try:
        for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
            # Store info for later processing to avoid issues with iterator invalidation
            try:
                processes_to_check.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue # Process disappeared or we lack permissions
            except Exception as iter_err:
                logging.error(f"Error collecting info for process {proc.pid if hasattr(proc, 'pid') else 'N/A'}: {iter_err}")
    except Exception as scan_err:
        logging.error(f"Error during initial process scan: {scan_err}", exc_info=True)
        return # Abort if scan fails

    # --- Second pass: Identify and terminate orphans and their children --- 
    pids_killed_this_run = set()
    for proc_info in processes_to_check:
        pid = proc_info.get('pid')
        ppid = proc_info.get('ppid')
        cmdline = proc_info.get('cmdline')
        name = proc_info.get('name', '').lower()

        if pid is None or ppid is None or not cmdline or pid in pids_killed_this_run:
            continue

        is_orphan = ppid == 1
        is_python = 'python' in name
        is_vllm_main = 'vllm.entrypoints.openai.api_server' in " ".join(cmdline)
        is_mp_helper = any('multiprocessing.spawn' in arg or 'multiprocessing.resource_tracker' in arg for arg in cmdline)

        # Target 1: Orphaned VLLM main process
        if is_orphan and is_python and is_vllm_main:
            logging.warning(f"Found potential orphaned VLLM main process: PID={pid}, Cmd='{' '.join(cmdline)}'. Attempting termination of process tree.")
            if _terminate_process_tree(pid):
                pids_killed_this_run.add(pid)
                # Also add potential children PIDs to avoid re-checking them (though _terminate_process_tree handles them)
                try:
                    # Re-fetch process to get children after termination attempt
                    # This part might be tricky due to timing
                    temp_proc = psutil.Process(pid) # Will raise NoSuchProcess if killed
                    children = temp_proc.children(recursive=True)
                    for child in children:
                         pids_killed_this_run.add(child.pid)
                except psutil.NoSuchProcess:
                    pass # Parent gone, assume children handled or will be found later if orphaned
                except Exception as e:
                     logging.warning(f"Could not reliably get/add children of killed orphan {pid}: {e}")

        # Target 2: Orphaned multiprocessing helpers (whose parent might have died *before* this check)
        elif is_orphan and is_python and is_mp_helper:
            # Check if it was already killed as part of a main process tree
            if pid not in pids_killed_this_run:
                logging.warning(f"Found potential orphaned VLLM multiprocessing helper: PID={pid}, Cmd='{' '.join(cmdline)}'. Attempting termination.")
                # For helpers, usually no deep process tree, just terminate the process itself
                if _terminate_process_tree(pid):
                     pids_killed_this_run.add(pid)

    if pids_killed_this_run:
         logging.info(f"Orphan process cleanup scan at startup finished. Terminated PIDs: {pids_killed_this_run}")
    else:
         logging.info("No suspected orphaned VLLM-related processes found during startup scan.")
