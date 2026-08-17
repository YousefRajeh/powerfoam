"""Phase 1 (powerfoam env): render Feature Foam's attention maps for every labeled
figurines frame. Run with D:\\conda\\envs\\powerfoam\\python.exe."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Downloads\claude_logs")
from featurefoam_lerf_bridge import render_scene_for_lerf_eval

SCENE = "figurines"
RESULT_FOLDER = Path(rf"D:\Downloads\claude_logs\_bridge_test_{SCENE}\rendered")

if RESULT_FOLDER.parent.exists():
    shutil.rmtree(RESULT_FOLDER.parent)

render_scene_for_lerf_eval(SCENE, RESULT_FOLDER)
print("PHASE1_DONE")
