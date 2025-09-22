import subprocess
import time
import os
import sys
import signal
import ctypes
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Logic for setting PDEATHSIG (adapted from vllm_loader.py) ---
IS_LINUX = platform.system() == "Linux"
PR_SET_PDEATHSIG = 1 # Linux specific constant
libc = None
prctl_syscall = None
can_use_prctl = False

if IS_LINUX:
    try:
        # Find libc path (handle different architectures)
        libc_path = None
        # Prioritize more specific paths first
        for path in ["/lib/x86_64-linux-gnu/libc.so.6", "/lib/aarch64-linux-gnu/libc.so.6", "/lib64/libc.so.6", "/lib/libc.so.6"]:
             if os.path.exists(path):
                 libc_path = path
                 break
        if libc_path:
            libc = ctypes.CDLL(libc_path)
            prctl_syscall = libc.prctl
            prctl_syscall.argtypes = [ctypes.c_int, ctypes.c_ulong]
            prctl_syscall.restype = ctypes.c_int
            can_use_prctl = True
            logging.info(f"Successfully loaded libc ({libc_path}) and prctl for PDEATHSIG.")
        else:
            logging.warning("Could not find libc.so.6 in common paths. PDEATHSIG functionality disabled.")

    except OSError as e:
        logging.warning(f"Could not load libc or find prctl: {e}. PDEATHSIG functionality disabled.")
        libc = None
        prctl_syscall = None
    except AttributeError:
        # Handle cases where prctl might not be an attribute even if libc loads
        logging.warning("libc loaded, but prctl attribute not found. PDEATHSIG functionality disabled.")
        libc = None
        prctl_syscall = None
        can_use_prctl = False
else:
    logging.info("Not running on Linux. PDEATHSIG functionality is not available.")

def set_pdeathsig_kill():
    """
    Sets the parent death signal to SIGKILL for the current process (Linux only).
    To be used as preexec_fn in subprocess.Popen.

    Input: None
    Output: None
    """
    # This function runs *in the child process* before exec.
    if IS_LINUX and can_use_prctl and prctl_syscall is not None:
        try:
            # Set the process to receive SIGKILL when the parent terminates
            ret = prctl_syscall(PR_SET_PDEATHSIG, signal.SIGKILL)
            if ret != 0:
                # Getting errno might be more informative, but print simply here
                # Use print as logging might not be configured yet or go to a different place
                print(f"[Child Process PID:{os.getpid()}] Warning: prctl(PR_SET_PDEATHSIG, SIGKILL) failed with return code {ret}", file=sys.stderr)
            # else:
                # Keep this commented out to reduce noise, uncomment for debugging
                # print(f"[Child Process PID:{os.getpid()}] Successfully set PDEATHSIG to SIGKILL.", file=sys.stderr)
        except Exception as e:
            print(f"[Child Process PID:{os.getpid()}] Exception calling prctl: {e}", file=sys.stderr)
    # else:
        # Keep this commented out to reduce noise
        # print(f"[Child Process PID:{os.getpid()}] PDEATHSIG not set (Not Linux or prctl unavailable).", file=sys.stderr)
# --- End PDEATHSIG Logic ---

# Number of child processes to start
NUM_CHILDREN = 3

def main():
    parent_pid = os.getpid()
    logging.info(f"Parent process PID: {parent_pid}")

    # Command for the child process: sleep for 30 seconds then exit
    # Use full path to sleep if necessary, though usually it's in PATH
    child_command = ["/bin/sleep", "30"]
    logging.info(f"Child command template: {' '.join(child_command)}")

    preexec_function = None
    if IS_LINUX and can_use_prctl:
        preexec_function = set_pdeathsig_kill
        logging.info("Will use preexec_fn to set PDEATHSIG on Linux.")
    else:
        logging.info("Will not use preexec_fn (Not Linux or prctl unavailable).")

    processes = [] # List to store Popen objects
    child_pids = [] # List to store child PIDs

    try:
        logging.info(f"Starting {NUM_CHILDREN} child processes...")
        for i in range(NUM_CHILDREN):
            try:
                process = subprocess.Popen(
                    child_command,
                    preexec_fn=preexec_function,
                    stdout=subprocess.PIPE, # Capture child stdout/stderr
                    stderr=subprocess.PIPE
                )
                processes.append(process)
                child_pids.append(process.pid)
                logging.info(f"  Child process {i+1}/{NUM_CHILDREN} started with PID: {process.pid}")
                # Give a tiny pause between starts if needed
                # time.sleep(0.1)
            except Exception as start_err:
                 logging.error(f"Failed to start child process {i+1}: {start_err}")
                 # Optionally, decide whether to continue starting others or stop

        if not child_pids:
            logging.error("No child processes were successfully started. Exiting.")
            return

        logging.info(f"All requested child processes started. PIDs: {child_pids}")

        logging.info("---------------------------------------------------------------------")
        logging.info(f"TEST INSTRUCTIONS (Linux Only with prctl):")
        logging.info(f"1. Open another terminal.")
        logging.info(f"2. Kill the parent process using: kill {parent_pid}")
        logging.info(f"3. Observe if ALL child processes (PIDs: {child_pids}) terminate quickly.")
        logging.info(f"   (Check with 'ps aux | grep sleep' or similar in the other terminal)")
        logging.info(f"4. Restart the script and repeat steps 2 & 3 using: kill -9 {parent_pid}")
        logging.info(f"If PDEATHSIG is working, ALL children should terminate almost immediately after the parent.")
        logging.info(f"If not killed manually, the parent script will wait ~40s for children and then exit.")
        logging.info("---------------------------------------------------------------------")

        # Wait longer than the child sleep time to allow for manual testing
        wait_time = 40
        logging.info(f"Parent sleeping for {wait_time} seconds while children run...")
        # We don't wait actively here using process.wait() or communicate()
        # because we want the script to keep running so it can be killed.
        # Instead, we just sleep.
        time.sleep(wait_time)
        logging.info("Parent finished sleeping.")

        # Check status of children after waiting (they should ideally have finished by now)
        logging.info("Checking status of child processes after wait period...")
        active_children = []
        for i, process in enumerate(processes):
             pid = child_pids[i]
             return_code = process.poll() # Check if finished without waiting
             if return_code is None:
                 logging.warning(f"Child {pid} still running after {wait_time}s! Attempting termination.")
                 active_children.append(process)
                 try:
                     process.terminate()
                     # Don't wait indefinitely here, give short timeout
                     process.wait(timeout=2)
                     return_code = process.poll()
                     if return_code is None:
                        logging.warning(f"Child {pid} did not terminate after SIGTERM, sending SIGKILL.")
                        process.kill()
                        process.wait(timeout=2)
                        return_code = process.poll()
                     logging.info(f"Child {pid} terminated by parent, final code: {return_code}")
                 except Exception as e:
                     logging.error(f"Error terminating child {pid} after wait: {e}")
                     try: # Final attempt to kill
                          process.kill()
                     except: pass
             else:
                 logging.info(f"Child {pid} finished with return code: {return_code}")
                 # Optionally log stdout/stderr here if needed
                 # stdout, stderr = process.communicate()

    except Exception as e:
        logging.error(f"An error occurred in the parent script: {e}", exc_info=True)

    except KeyboardInterrupt:
         logging.info("\nParent script interrupted (Ctrl+C).")
         # Fall through to finally block for cleanup

    finally:
        # Cleanup running child processes if the script exits for any reason
        # (like KeyboardInterrupt or normal completion after sleep)
        logging.info("Starting cleanup in finally block...")
        if 'processes' in locals() and processes:
            running_pids = [p.pid for p in processes if p.poll() is None]
            if running_pids:
                logging.info(f"Attempting to terminate remaining child process(es)... PIDs: {running_pids}")
                for process in processes:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except Exception as term_err:
                            logging.warning(f"Error sending SIGTERM to child {process.pid}: {term_err}")
                # Wait briefly for termination
                time.sleep(0.5)
                for process in processes:
                    if process.poll() is None:
                        try:
                            logging.warning(f"Child {process.pid} did not terminate gracefully, killing...")
                            process.kill()
                            process.wait(timeout=1) # Short wait after kill
                        except Exception as kill_err:
                            logging.error(f"Error sending SIGKILL or waiting for child {process.pid}: {kill_err}")
                logging.info("Child process cleanup attempt finished.")
            else:
                logging.info("No running child processes found needing cleanup.")
        logging.info("Parent script finished.")


if __name__ == "__main__":
    # Create ./tests directory if it doesn't exist
    if not os.path.exists('./tests'):
        os.makedirs('./tests')
        print("Created ./tests directory.")
    main() 