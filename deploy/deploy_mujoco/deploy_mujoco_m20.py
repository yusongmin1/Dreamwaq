"""M20 MuJoCo Sim2Sim (assets from deploy_mujoco_m20)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from deploy_mujoco_base import MujocoDeploy

if __name__ == "__main__":
    MujocoDeploy(default_config="m20.yaml").run_sim()
