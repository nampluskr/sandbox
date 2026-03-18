import os
import sys
import subprocess
import time


SCRIPTS = [
    # "mvtec_11_dfm_resnet18.py",         # 완료
    # "mvtec_11_dfm_resnet50.py",         # 완료
    # "mvtec_12_dfkde_resnet18.py",       # 완료
    # "mvtec_12_dfkde_resnet50.py",       # 완료
    # "mvtec_13_ganomaly.py",             # 완료
    # "mvtec_14_fre_resnet18.py",         # 완료
    # "mvtec_14_fre_resnet50.py",         # 완료
    # "mvtec_15_draem.py",
    # "mvtec_17_supersimplenet.py",       # 완료
    # "mvtec_18_uninet.py",               # 완료
    # "mvtec_19_dinomaly_small.py",
    # "mvtec_19_dinomaly_base.py",
    # "mvtec_19_dinomaly_large.py",
    # "mvtec_20_anomalydino_small.py",
    # "mvtec_20_anomalydino_base.py",
    # "mvtec_20_anomalydino_large.py",
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
