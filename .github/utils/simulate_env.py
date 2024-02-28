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


def run_script(script_path, script_args, rank, world_size):
    proccess = subprocess.Popen(['python', script_path, script_args, "--rank", rank, "--world-size", world_size])
    return proccess


def wait_for_it(proccesses, timeout):
    try:
        proccesses[0].wait(timeout=timeout)
    except:
        for p in proccesses:
            p.kill()


def main():
    args = parser.parse_args()

    processes = []
    for rank in range(args.world_size):
        p = run_script(args.script_path, args.script_args, rank, args.world_size)
        processes.append(p)

    wait_for_it(processes, args.timeout)


if __name__ == "__main__":
    main()
