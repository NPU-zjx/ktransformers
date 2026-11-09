"""Host entry point. Edit base.sh for the experiment settings."""
import argparse
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).with_name("base.sh").resolve()


def invoke(action):
    subprocess.run(["bash", str(SCRIPT), action], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "serve", "rollout", "stop", "torch", "nsys"))
    action = parser.parse_args().action
    if action in ("run", "torch", "nsys"):
        prefix = "" if action == "run" else action + "-"
        invoke(prefix + "serve")
        try:
            invoke(prefix + "profile" if prefix else "rollout")
        finally:
            invoke(prefix + "stop")
    else:
        invoke(action)
