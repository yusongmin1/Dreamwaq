import os
import time

import mujoco.viewer
import mujoco
import numpy as np
import yaml
import onnxruntime as ort
from pynput import keyboard

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def resolve_config_path(config_file):
    if os.path.isabs(config_file) and os.path.isfile(config_file):
        return config_file
    candidates = [
        config_file,
        os.path.join(CONFIG_DIR, config_file),
        os.path.join(REPO_ROOT, config_file),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Config file not found: {config_file}")

GEAR_SPEEDS = {1: 0.5, 2: 1.0, 3: 1.5}

x_vel_cmd, y_vel_cmd, yaw_vel_cmd, stand_command = 0.05, 0.0, 0.0, 0.0
y_vel_max, yaw_vel_max = 1.0, 3.0
gear = 1
x_vel_max = GEAR_SPEEDS[gear]
cmd = [x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
joystick_use = True
joystick_opened = False
def on_press(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, stand_command, cmd, gear, x_vel_max
    try:
        if key.char == '6':
            x_vel_cmd = min(x_vel_cmd + 0.1, x_vel_max)
        elif key.char == '7':
            x_vel_cmd = max(x_vel_cmd - 0.1, -x_vel_max)
        elif key.char == '8':
            y_vel_cmd += 0.3
        elif key.char == '9':
            y_vel_cmd -= 0.3
        elif key.char == '-':
            yaw_vel_cmd += 0.5
        elif key.char == '=':
            yaw_vel_cmd -= 0.5
        elif key.char == 'c':
            stand_command = 0
        elif key.char == 'v':
            stand_command = 1
        elif key.char == '1':
            x_vel_cmd = 0
            y_vel_cmd = 0
            yaw_vel_cmd = 0
        elif key.char == '2':
            gear = 1
            x_vel_max = GEAR_SPEEDS[gear]
            x_vel_cmd = max(min(x_vel_cmd, x_vel_max), -x_vel_max)
            print(f"Gear {gear}: max vx = {x_vel_max:.1f} m/s")
        elif key.char == '3':
            gear = 2
            x_vel_max = GEAR_SPEEDS[gear]
            x_vel_cmd = max(min(x_vel_cmd, x_vel_max), -x_vel_max)
            print(f"Gear {gear}: max vx = {x_vel_max:.1f} m/s")
        elif key.char == '4':
            gear = 3
            x_vel_max = GEAR_SPEEDS[gear]
            x_vel_cmd = max(min(x_vel_cmd, x_vel_max), -x_vel_max)
            print(f"Gear {gear}: max vx = {x_vel_max:.1f} m/s")
        else:
            return
        cmd = [x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
        print(
            f"Updated velocities: vx={x_vel_cmd}, vy={y_vel_cmd}, "
            f"dyaw={yaw_vel_cmd} stand_command={stand_command} gear={gear}"
        )
    except AttributeError:
        pass

class Deploy:
    def __init__(self):
        import argparse
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-c", "--config_file", type=str,
                         default="go2.yaml",
                         help="config file name in the config folder")
        self.args = self.parser.parse_args()
        self.config_file = resolve_config_path(self.args.config_file)
        with open(self.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = resolve_repo_path(config["policy_path"])
            self.xml_path = resolve_repo_path(config["xml_path"])
            print(f"config: {self.config_file}")
            print(f"xml: {self.xml_path}")
            print(f"policy: {self.policy_path}")

            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.torque_limits = np.array(config["torque_limits"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.cmd = np.array(config["cmd_init"], dtype=np.float32)

        # define context variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.obs_hist_buf = np.zeros(self.num_obs * 6, dtype=np.float32)

        self.dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.line_vel = np.zeros(3, dtype=np.float32)

        self.counter = 0

        # Load robot model
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt

        # load policy
        # self.policy = torch.jit.load(self.policy_path)
        self.policy = ort.InferenceSession(self.policy_path, provifers=["CPUExecutionProvider"])
    
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation
    
    def get_robot_state(self):
        self.dof_pos = self.d.qpos[7:]
        self.dof_vel = self.d.qvel[6:]
        self.quat = self.d.qpos[3:7]
        self.ang_vel = self.d.qvel[3:6]
        self.line_vel = self.d.qvel[3:6]
    
    def compute_torques(self, actions):
        dof_err = self.default_angles - self.dof_pos # 各DOF默认位置 - 目前各DOF位置

        actions_scaled = actions * self.action_scale # action * 0.25
        
        torques = self.kps * (
                actions_scaled + dof_err
            ) + self.kds * (- self.dof_vel)
        
    
        return np.clip(torques, -self.torque_limits, self.torque_limits)
    
    def compute_observation(self):
        dof_error = (self.dof_pos - self.default_angles) * self.dof_pos_scale

        dof_vel = self.dof_vel * self.dof_vel_scale
        gravity_orientation = self.get_gravity_orientation(self.quat)
        omega = self.ang_vel * self.ang_vel_scale
        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = cmd * self.cmd_scale 
        self.obs[9 : 9 + self.num_actions] = dof_error
        self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dof_vel
        self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] =self.action 
        self.obs[45]=stand_command
        self.obs_hist_buf = self.obs_hist_buf[46:]
        self.obs_hist_buf = np.concatenate((self.obs_hist_buf, self.obs), axis=-1)


    def run_sim(self):
        self.vel_data = []
        self.cmd_data = []
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            
            while viewer.is_running() and time.time() - start < self.simulation_duration:
                step_start = time.time()
                tau = self.compute_torques(self.action)
                self.d.ctrl[:] = tau
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(self.m, self.d)

                self.counter += 1
                if self.counter % self.control_decimation == 0:
                    self.get_robot_state()
                    self.compute_observation()
                    
                    out_name = self.policy.get_outputs()[0].name
                    actions = self.policy.run([out_name], {"obs": self.obs_hist_buf.reshape(1, -1)})
                    self.action = actions[0][0]

                    self.vel_data.append(self.dof_pos.copy())
                    
                viewer.sync()
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    deploy = Deploy()

    deploy.run_sim()
