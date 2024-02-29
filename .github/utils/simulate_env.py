import os
import shlex
import subprocess
from argparse import ArgumentParser


parser = ArgumentParser("run SAR locally")

parser.add_argument("--script-path", type=str, required=True,
        help="path to the script, which will be runned")

parser.add_argument("--timeout", type=int, required=True,
        help="timeout after which processes will be killed")

parser.add_argument("--script-args", type=str, required=True,
        help="arguments for the sccripts (without world size and rank)")

parser.add_argument("--world-size", type=int, required=True,
        help="Number of processes")


def run_script(parsed_command):
    proccess = subprocess.Popen(parsed_command)
    return proccess


def wait_for_it(proccesses, timeout):
    try:
        proccesses[0].wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        for p in proccesses:
            p.kill()
        raise RuntimeError("Script didn't finish - timeout mechanism was triggered")

    for p in proccesses:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"There has been an exception in the runned script - return code: {p.returncode}")
     

def parse_command(script_path, script_args, rank, world_size):
    script_args += f" --rank {str(rank)} --world-size {str(world_size)}"
    parsed_command = shlex.split(script_args)
    parsed_command.insert(0, script_path)
    parsed_command.insert(0, "python")
    
    ip_path_index = parsed_command.index("--ip-file") + 1
    parsed_command[ip_path_index] = os.path.expanduser(parsed_command[ip_path_index])
    
    partition_path_index = parsed_command.index("--partitioning-json-file") + 1
    parsed_command[partition_path_index] = os.path.expanduser(parsed_command[partition_path_index])
    
    return parsed_command



def main():
    args = parser.parse_args()
    
    processes = []
    for rank in range(args.world_size):
        parsed_command = parse_command(args.script_path, args.script_args, rank, args.world_size)
        p = run_script(parsed_command)
        processes.append(p)

    wait_for_it(processes, args.timeout)


if __name__ == "__main__":
    main()
