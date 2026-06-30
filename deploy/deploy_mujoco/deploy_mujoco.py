"""M20 Big MuJoCo Sim2Sim."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from deploy_mujoco_base import MujocoDeploy

if __name__ == "__main__":
    MujocoDeploy(default_config="m20_big.yaml").run_sim()
