import os
import sys
import subprocess
import time


SCRIPTS = [
    "mvtec_01_stfpm_resnet18.py",
    "mvtec_01_stfpm_resnet50.py",
    "mvtec_02_reversedistill_resnet18.py",
    "mvtec_02_reversedistill_wideresnet50.py",
    "mvtec_03_efficientad_small.py",
    "mvtec_03_efficientad_medium.py",
    "mvtec_05_fastflow_resnet18.py",
    "mvtec_05_fastflow_wideresnet50.py",
    "mvtec_05_fastflow_deit.py",          
    "mvtec_05_fastflow_cait.py",
    "mvtec_06_csflow_efficientnetb5.py",
    "mvtec_07_uflow_resnet18.py",
    "mvtec_07_uflow_wideresnet50.py",
    "mvtec_07_uflow_mcait.py",
    "mvtec_08_patchcore_resnet18.py",
    "mvtec_08_patchcore_wideresnet50.py",
    "mvtec_09_padim_resnet18.py",
    "mvtec_09_padim_wideresnet50.py",
    "mvtec_10_cfa_wideresnet50.py",
    "mvtec_04_cflow_resnet18.py",
    "mvtec_04_cflow_wideresnet50.py",
]

if __name__ == "__main__":
    success_list = []
    failure_list = []

    for idx, script in enumerate(SCRIPTS, start=1):
        if not os.path.exists(script):
            print(f"[NOT FOUND] {script}")
            failure_list.append(script)
            continue

        print("\n" + "=" * 70)
        print(f"[Running ({idx}/{len(SCRIPTS)})] {script}")
        print("=" * 70)

        start_time = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script],
                stdout=sys.stdout,  # Directly stream to console
                stderr=sys.stderr,  # Directly stream to console
                # timeout=1800        # 30 minutes timeout
            )
            elapsed = time.time() - start_time
            hours, remainders = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainders, 60)
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            if result.returncode == 0:
                print(f"\n[SUCCESS] {script}: {elapsed_str}")
                success_list.append(script)
            else:
                print(f"\n[FAILED] {script} {elapsed_str}")
                failure_list.append(script)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"[TIMEOUT] {script} (killed after 30 minutes)")
            failure_list.append(script)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[EXCEPTION] {script}: {e}")
            failure_list.append(script)

    print("\n" + "=" * 70)
    print("[FINISHED] Execution Summary")
    print(f"Success: {len(success_list)}")
    for filename in success_list:
        print(f" > {filename}")
    print(f"Failure: {len(failure_list)}")
    for filename in failure_list:
        print(f" > {filename}")
