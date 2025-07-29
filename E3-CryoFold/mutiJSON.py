import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_produce_json(script_path, folder_path, folder_name):
    pdb_file = os.path.join(folder_path, f"{folder_name}.pdb")
    json_output = os.path.join(folder_path, "seq_chain_info.json")

    if not os.path.exists(pdb_file):
        return f"❌ Skipped: PDB not found: {pdb_file}"

    try:
        subprocess.run(['python', script_path, pdb_file, json_output], check=True)
        return f"✅ Success: {folder_name}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error: {folder_name} - {e}"

def run_all_parallel(base_dir, script_path="/ziyingz/Programs/E3-CryoFold/produce_JSON.py", max_workers=15):
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            tasks.append(executor.submit(run_produce_json, script_path, folder_path, folder))

        for future in as_completed(tasks):
            print(future.result())

# 执行任务
run_all_parallel("/ziyingz/library/cryodata/valdata")
