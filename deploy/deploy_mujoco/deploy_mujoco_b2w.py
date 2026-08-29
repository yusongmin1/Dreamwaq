"""B2W MuJoCo Sim2Sim (mjcf from rl_sar b2w_description)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from deploy_mujoco_base import MujocoDeploy

if __name__ == "__main__":
    MujocoDeploy(default_config="b2w.yaml").run_sim()
