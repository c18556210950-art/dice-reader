"""Independent, local-only dice annotation and training tools."""
import os
from pathlib import Path

# Separate project settings from any user HUB login and prevent telemetry/version probes.
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[1]/'.runtime'/'dice-training'/'yolo-config')
