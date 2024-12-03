import subprocess
from tqdm import tqdm


for i in tqdm(range(30), desc="Running iterations", unit="iteration"):
    process = subprocess.Popen([
        "python3", "main.py",
        "--iteration", str(i)
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # 等待进程结束
    process.communicate()