"""M20 MuJoCo Sim2Sim for CTS policy (policy_cts.onnx, student encoder + actor)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from deploy_mujoco_base import MujocoDeploy

if __name__ == "__main__":
    MujocoDeploy(default_config="m20_cts.yaml").run_sim()
