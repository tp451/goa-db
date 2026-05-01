"""Step 1: Train row segmentation model. Wrapper around train_yolo.py."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "train_yolo.py", "--task", "seg"] + sys.argv[1:]))
