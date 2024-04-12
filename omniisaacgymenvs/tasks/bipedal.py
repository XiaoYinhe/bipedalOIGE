
import math

import numpy as np
import torch
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.articulations.bipedal import Bipedal
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.replicator.isaac")
# import omni.replicator.core as oc
# import omni.replicator.isaac as dr
# import omni.replicator.core as rep



class BipedalTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)

        self._num_observations = 30
        # 六个电机输出
        self._num_actions = 6

        RLTask.__init__(self, name, env)


        self.calt_cnt =-100
        self.rwd_value_wh =0
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "lin_vel_xy": torch_zeros(),
            "lin_vel_z": torch_zeros(),
            "ang_vel_z": torch_zeros(),
            "joint_acc": torch_zeros(),
            "action_rate": torch_zeros(),
        }
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]

        
        # self._robot_translation = torch.tensor([0.0, 0.0, 0.1884+0.01])
        # quat_ang = quat_from_euler_xyz(torch.tensor(0),torch.tensor(0.568),torch.tensor(0))


        self._robot_translation = torch.tensor([0.0, 0.0, 0.193+0.03])
        quat_ang = quat_from_euler_xyz(torch.tensor(0),torch.tensor(0),torch.tensor(0))
        
        self._robot_orientation = quat_ang
        
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

    def set_up_scene(self, scene) -> None:

        self.get_robot()
        # 复制场景
        super().set_up_scene(scene)

        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Robot", name="robot_view", reset_xform_properties=False
        )
        scene.add(self._robots)


        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)
        return

     # 获取机器人场景
    def get_robot(self):
        
        robot = Bipedal(
            prim_path=self.default_zero_env_path + "/Robot", name="Robot", translation=self._robot_translation,
            orientation=self._robot_orientation
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Robot", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("Robot")
        )

          # Configure joint properties
        hip_joint_paths = ["lhipJoint","rhipJoint"]
        for joint_path in hip_joint_paths:
            set_drive(f"{robot.prim_path}/{joint_path}", "angular", "position", 0, self.Kp, self.Kd, 150)

        ctrl_joint_paths = ["lf1Joint","lb1Joint","rf1Joint","rb1Joint"]
        for joint_path in ctrl_joint_paths:
            set_drive(f"{robot.prim_path}/{joint_path}", "angular", "position", 0, self.Kp, self.Kd, 40)

        # other_joint_paths = ["lf2Joint","lb2Joint","lfootJoint","rf2Joint","rb2Joint","rfootJoint"]
        # for joint_path in other_joint_paths:
        #     set_drive(f"{robot.prim_path}/{joint_path}", "angular", "position", 0, 0, 0, 0)


    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        root_velocities = self._robots.get_velocities(clone=False)
        dof_pos = self._robots.get_joint_positions(joint_indices = self.ctrl_dof_idx, clone=False)
        dof_vel = self._robots.get_joint_velocities(joint_indices = self.ctrl_dof_idx, clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]
        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale

        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )
        # print(acc)
        obs = torch.cat(
            (
                base_lin_vel,   
                base_ang_vel,
                # lin_acc_w,
                # ang_acc_w,
                projected_gravity,
                commands_scaled,
                dof_pos * self.dof_pos_scale,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        
        self.obs_buf[:] = obs

        observations = {self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # env * 6
        self.actions[:] = actions.clone().to(self._device)

        current_targets = torch.zeros_like(self.actions)
        current_targets = self.current_targets +  self.action_scale * self.actions * self.dt
        
        self.current_targets[:] = tensor_clamp(
            current_targets, self.anymal_dof_lower_limits, self.anymal_dof_upper_limits
        )
        
        # current_targets[reset_env_ids] = 0

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        testSetPos = torch.ones_like(self.current_targets) * 0.5

        self._robots.set_joint_position_targets(self.current_targets, indices,self.ctrl_dof_idx)

        reset_buf = self.reset_buf.clone()
        if self._dr_randomizer.randomize:
            rand_envs = torch.where(
                self.randomization_buf >= self._dr_randomizer.min_frequency,
                torch.ones_like(self.randomization_buf),
                torch.zeros_like(self.randomization_buf),
            )
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            self.dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        dof_pos = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        self.current_targets[env_ids] = dof_pos[:,self.ctrl_dof_idx]

        # lin_vel and ang_vel
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._robots.set_joint_positions(dof_pos, indices)
        self._robots.set_joint_velocities(dof_vel, indices)


        # self._robot_translation = torch.tensor([0.0, 0.0, 0.1923+0.01])
        # quat_ang = quat_from_euler_xyz(torch.tensor(0),torch.tensor(0),torch.tensor(0))
        
        self._robots.set_world_poses(
            self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices
        )
        self._robots.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.last_vel = torch.zeros(self._num_envs, 6, dtype=torch.float, device=self._device)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def post_reset(self):
        ctrl_dof_paths = ["lhipJoint","lf1Joint","lb1Joint","rhipJoint","rf1Joint","rb1Joint"]
        self.ctrl_dof_idx = torch.tensor(
            [self._robots._dof_indices[j] for j in ctrl_dof_paths], device=self._device, dtype=torch.long
        )
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()

        dof_limits = self._robots.get_dof_limits().to(self._device)
        print(dof_limits.size())
        dof_limits_sel = dof_limits[:, self.ctrl_dof_idx]
        print(dof_limits_sel.size())
        self.anymal_dof_lower_limits = dof_limits_sel[0, :, 0].to(device=self._device)
        self.anymal_dof_upper_limits = dof_limits_sel[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(
            (self._num_envs, 6), dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )

        self.current_targets =  torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.last_vel = torch.zeros(self._num_envs, 6, dtype=torch.float, device=self._device)
  
        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._robots.get_world_poses(clone=False)
        root_velocities = self._robots.get_velocities(clone=False)
        dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)

        dof_vel_sel = dof_vel[:,self.ctrl_dof_idx]

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel_sel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        # rwd_height = 0.01 * torch.exp(torso_position[:,2])
        # rwd_height += 0.01 * torch.exp(-torch.abs(projected_gravity[:,0]))
        # rwd_height += 0.01 * torch.exp(-torch.abs(projected_gravity[:,1]))
        
        # rew_cosmetic = (
        #     torch.sum(torch.abs(dof_pos[:, 0:4]), dim=1) * self.rew_scales["cosmetic"]
        # )

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        
        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_joint_acc + rew_action_rate  + rew_lin_vel_z #+rwd_height
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel_sel[:]

        # print(projected_gravity)

        fall_side1 = torch.where(torch.abs(projected_gravity[:,0]) > torso_position[:, 2], 1, 0) 
        fall_side1 = torch.where(torch.abs(projected_gravity[:,1]) > torso_position[:, 2], 1, fall_side1)
        self.fallen_over = self.is_base_below_threshold(threshold=0.19, ground_heights=0.0) | fall_side1
        # print(torch.sum(total_reward).to(torch.float))
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()
        self.randomization_buf += 1

    def is_base_below_threshold(self, threshold, ground_heights):
        base_pos, _ = self._robots.get_world_poses()
        base_heights = base_pos[:, 2]
        base_heights -= ground_heights
        return (base_heights[:] < threshold) | (base_heights[:] > 0.5)

    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
