"""Step 3: Train header detection model. Wrapper around train_yolo.py."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "train_yolo.py", "--task", "detect"] + sys.argv[1:]))
