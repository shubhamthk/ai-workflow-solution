import os, sys, subprocess
os.environ["TEST_MODE"] = "1"
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))
