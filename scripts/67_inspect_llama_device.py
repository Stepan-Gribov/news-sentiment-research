import json
import pathlib
import shutil
import subprocess

server = pathlib.Path("./local/AppData/Local/Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe")
for args in [[str(server), "--list-devices"], [str(server), "--version"], ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=30)
        print(json.dumps({"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=True), flush=True)
    except Exception as error:
        print(json.dumps({"error": str(error), "command": args}), flush=True)
