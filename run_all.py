import subprocess
import sys
import time

def run_step(command_list, step_name):
    print(f"\n==================================================")
    print(f"STARTING STEP: {step_name}")
    print(f"Command: {' '.join(command_list)}")
    print(f"==================================================")
    start_time = time.time()
    
    # Run process and stream output to console
    process = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output line by line as it is produced
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
        
    process.wait()
    duration = time.time() - start_time
    
    if process.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {process.returncode} after {duration:.2f}s!")
        sys.exit(process.returncode)
    else:
        print(f"\n[SUCCESS] Step '{step_name}' completed in {duration:.2f}s.")

def main():
    steps = [
        (["python", "scripts/download_data.py", "--config", "configs/experiments/main.yaml"], "1. Download Data"),
        (["python", "scripts/build_dataset.py", "--config", "configs/experiments/main.yaml"], "2. Build Dataset"),
        (["python", "scripts/run_experiment.py", "--config", "configs/experiments/main.yaml"], "3. Run Out-of-Sample Experiments"),
        (["python", "scripts/run_robustness_gap.py"], "4. Run Robustness Gap Analysis"),
        (["python", "scripts/make_tables.py", "--config", "configs/experiments/main.yaml"], "5. Generate Tables"),
        (["python", "scripts/make_figures.py", "--config", "configs/experiments/main.yaml"], "6. Generate Figures"),
    ]
    
    total_start = time.time()
    for cmd, name in steps:
        run_step(cmd, name)
        
    total_duration = time.time() - total_start
    print(f"\n==================================================")
    print(f"ALL STEPS COMPLETED SUCCESSFULLY IN {total_duration:.2f}s!")
    print(f"==================================================")

if __name__ == "__main__":
    main()
