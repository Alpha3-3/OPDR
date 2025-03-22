import subprocess
import os

def run_programs(program_list):
    for program in program_list:
        if not os.path.isfile(program):
            print(f"Error: {program} not found. Skipping...")
            continue

        print(f"Running {program} ...")
        # Run the program and wait for it to complete
        result = subprocess.run(['python', program], capture_output=True, text=True)

        # Print the output and any errors
        print(f"Output of {program}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors from {program}:\n{result.stderr}")
        print(f"{program} completed with return code {result.returncode}\n")

if __name__ == '__main__':
    # Define the list of Python scripts to run
    programs = [
        'Parallel CIFAR-10.py',
        'Parallel Isolet.py',
        'Parallel Fasttext.py',
        'Parallel PBMC3k'
    ]
    run_programs(programs)
