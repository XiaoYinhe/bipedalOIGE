
import math

import numpy as np
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.articulations.bipedal import Bipedal
from omniisaacgymenvs.robots.articulations.views.bipedal_view import BipedalView

from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from pxr import UsdLux, UsdPhysics


class BipedalTerrainTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.height_samples = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0

        self._num_observations = 30
        self._num_actions = 6

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        # self.height_points = self.init_height_points()
        
        # joint positions offsets
        self.default_dof_pos = torch.zeros(
            (self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False
        )
        # reward episode sums
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "lin_vel_xy": torch_zeros(),
            "lin_vel_z": torch_zeros(),
            "ang_vel_z": torch_zeros(),
            "ang_vel_xy": torch_zeros(),
            "orient": torch_zeros(),
            "torques": torch_zeros(),
            "joint_acc": torch_zeros(),
            "base_height": torch_zeros(),
            "air_time": torch_zeros(),
            "collision": torch_zeros(),
            "stumble": torch_zeros(),
            "action_rate": torch_zeros(),
            "hip": torch_zeros(),
            "step_pre": torch_zeros(),
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
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self._task_cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["contact_pre"] = self._task_cfg["env"]["learn"]["stepFreRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        # self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp_base = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd_base = self._task_cfg["env"]["control"]["damping"]
        self.maxtor_base = self._task_cfg["env"]["control"]["maxtorque"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1

        # random
        self.kp_rand = self._task_cfg["env"]["control"]["kpRand"]
        self.kd_rand = self._task_cfg["env"]["control"]["kdRand"]
        self.maxtor_rand = self._task_cfg["env"]["control"]["maxtorqueRand"]
        self.out_rand = self._task_cfg["env"]["control"]["outRand"]
        self.def_pos_rand = self._task_cfg["env"]["control"]["defPosRand"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]

        self._task_cfg["sim"]["default_physics_material"]["static_friction"] = self._task_cfg["env"]["terrain"][
            "staticFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["dynamic_friction"] = self._task_cfg["env"]["terrain"][
            "dynamicFriction"
        ]
        self._task_cfg["sim"]["default_physics_material"]["restitution"] = self._task_cfg["env"]["terrain"][
            "restitution"
        ]

        self._task_cfg["sim"]["add_ground_plane"] = False

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self._task_cfg["env"]["learn"]["addNoise"]
        noise_level = self._task_cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self._task_cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self._task_cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self._task_cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12:18] = self._task_cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[18:24] = self._task_cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[24:30] = 0.0  # previous actions
        return noise_vec


    def _create_trimesh(self, create_mesh=True):
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_robot()
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
        
        self._robots = BipedalView(
            prim_paths_expr="/World/envs/.*/Robot", name="robot_view",track_contact_forces=True
        )
        scene.add(self._robots)
        scene.add(self._robots._knees)
        scene.add(self._robots._base)
        scene.add(self._robots._hip)
        scene.add(self._robots._feet)

    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("robot_view"):
            scene.remove_object("robot_view", registry_only=True)

        if scene.object_exists("knees_view"):
            scene.remove_object("knees_view", registry_only=True)

        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)

        self._robots = BipedalView(
            prim_paths_expr="/World/envs/.*/Robot", name="robot_view",track_contact_forces=True
        )
        scene.add(self._robots)
        scene.add(self._robots._knees)
        scene.add(self._robots._base)
        scene.add(self._robots._hip)
        scene.add(self._robots._feet)

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum:
            self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(
            0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
        )
        self.terrain_types = torch.randint(
            0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
        )
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_robot(self):
        init_translation = torch.tensor([0.0, 0.0, 0.193+0.03])
        init_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        robot = Bipedal(
            prim_path=self.default_zero_env_path + "/Robot",
            name="Robot",
            translation=init_translation,
            orientation=init_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "bipedal", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("bipedal")
        )
        robot.set_anymal_properties(self._stage, robot.prim)
        robot.prepare_contacts(self._stage, robot.prim)


    def post_reset(self):
        ctrl_dof_paths = ["lhipJoint","lf1Joint","lb1Joint","rhipJoint","rf1Joint","rb1Joint"]
        self.ctrl_dof_idx = torch.tensor(
            [self._robots._dof_indices[j] for j in ctrl_dof_paths], device=self._device, dtype=torch.long
        )

        self.base_init_state = torch.tensor(
            self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.feet_contact_time = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.last_feet_touch_state = torch.zeros(self.num_envs, 2, dtype=torch.int, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device, requires_grad=False)

        self.num_dof = self._robots.num_dof
        self.dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        self.knee_pos = torch.zeros((self.num_envs * 4, 3), dtype=torch.float, device=self.device)
        self.knee_quat = torch.zeros((self.num_envs * 4, 4), dtype=torch.float, device=self.device)
        # PD param
        self.Kp = torch_rand_float(self.Kp_base-self.kp_rand, self.Kp_base+self.kp_rand, (self.num_envs,1), device=self.device)
        self.Kd = torch_rand_float(self.Kd_base-self.kd_rand, self.Kd_base+self.kd_rand, (self.num_envs,1), device=self.device)
        self.maxtor = torch_rand_float(self.maxtor_base-self.maxtor_rand, self.maxtor_base+self.maxtor_rand, (self.num_envs,1), device=self.device)
        self.def_pos = torch.zeros((self.num_envs, 6), device=self.device)
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        self.dof_vel[env_ids] = torch.zeros((len(env_ids), self._robots.num_dof), device=self._device)
        self.dof_pos_sel = self.dof_pos[:,self.ctrl_dof_idx]
        self.dof_vel_sel = self.dof_vel[:,self.ctrl_dof_idx]

        self.update_terrain_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.base_quat[env_ids] = self.base_init_state[3:7]
        self.base_velocities[env_ids] = self.base_init_state[7:]

        self._robots.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_positions(positions=self.dof_pos[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
            1
        )  # set small commands to zero

        # PD random
        self.Kp[env_ids] = torch_rand_float(self.Kp_base-self.kp_rand, self.Kp_base+self.kp_rand, (len(env_ids),1), device=self.device)
        self.Kd[env_ids] = 0.05*torch_rand_float(self.Kd_base-self.kd_rand, self.Kd_base+self.kd_rand, (len(env_ids),1), device=self.device)
        self.maxtor[env_ids] = torch_rand_float(self.maxtor_base-self.maxtor_rand, self.maxtor_base+self.maxtor_rand, (len(env_ids),1), device=self.device)
        self.def_pos[env_ids] = torch_rand_float(-self.def_pos_rand, self.def_pos_rand, (len(env_ids),6), device=self.device)

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        
        self.feet_contact_time[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # do not change on initial reset
            return
        root_pos, _ = self._robots.get_world_poses(clone=False)
        distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (
            distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25
        )
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def refresh_dof_state_tensors(self):
        self.dof_pos = self._robots.get_joint_positions(clone=False) 
        self.dof_vel = self._robots.get_joint_velocities(clone=False)
        self.dof_pos_sel = self.dof_pos[:,self.ctrl_dof_idx] - self.def_pos
        self.dof_vel_sel = self.dof_vel[:,self.ctrl_dof_idx]


    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_velocities = self._robots.get_velocities(clone=False)
        self.knee_pos, self.knee_quat = self._robots._knees.get_world_poses(clone=False)

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        self.actions = actions.clone().to(self.device)
        self.actions[:,0] = torch.clip(0.25*self.actions[:,0],-0.25,0.25)
        self.actions[:,3] = torch.clip(0.25*self.actions[:,3],-0.25,0.25)
        cols_to_clip = [1, 2, 4, 5]
        self.actions[:,cols_to_clip] = torch.clip(self.actions[:,cols_to_clip],-1,1)

        # print(self.def_pos)
        for i in range(self.decimation):
            if self.world.is_playing():
                # print(f"pos:{self.dof_pos_sel}")
                # print(self.Kd)
                motorTorqueClip = self.maxtor*torch.clip((30-torch.abs(self.dof_vel_sel))/30,0,1)
                # print(motorTorqueClip)
                torques = torch.clip(
                    self.Kp * (self.action_scale * self.actions + self.def_pos - self.dof_pos_sel)
                    - self.Kd * self.dof_vel_sel
                    +torch_rand_float(-self.out_rand, self.out_rand, (self.num_envs,6), device=self.device),
                    -motorTorqueClip,
                    motorTorqueClip,
                )
                # print(f"vel:{self.dof_vel_sel}")
                # print(f"torques:{torques}")
                self._robots.set_joint_efforts(torques, joint_indices = self.ctrl_dof_idx)
                self.torques = torques
                SimulationContext.step(self.world, render=False)
                self.refresh_dof_state_tensors()

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self.world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            self.common_step_counter += 1
            if self.common_step_counter % self.push_interval == 0:
                self.push_robots()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

            self.check_termination()
            self.get_states()
            self.calculate_metrics()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            if self.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel_sel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(
            -0.5, 0.5, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self._robots.set_velocities(self.base_velocities)

    def check_termination(self):
        self.timeout_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )
        knee_contact = torch.norm(self._robots._knees.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3), dim=-1)
        hip_contact =  torch.norm(self._robots._hip.get_net_contact_forces(clone=False).view(self._num_envs, 2, 3), dim=-1)
        base_contace = torch.norm(self._robots._base.get_net_contact_forces(clone=False), dim=1)
        total_force = torch.sum(knee_contact, dim=-1)+torch.sum(hip_contact, dim=-1)+base_contace
        # print(total_force)
        # print(torch.sum(base_contace, dim=-1))
        fall_side1 = torch.where(torch.abs(self.projected_gravity[:,0]) > 0.5, 1, 0) 
        fall_side1 = torch.where(torch.abs(self.projected_gravity[:,1]) > 0.5, 1, fall_side1)
        # print(torch.sum(hip_contact, dim=-1))
        self.has_fallen = ( total_force > 0.1)  | fall_side1

        self.reset_buf = self.has_fallen.clone()
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def calculate_metrics(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.base_pos[:, 2] - 0.52) * self.rew_scales["base_height"]

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel_sel), dim=1) * self.rew_scales["joint_acc"]

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        # action rate penalty
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )

        # cosmetic penalty for hip motion
        rew_hip = (
            torch.sum(torch.abs(self.dof_pos_sel), dim=1) * self.rew_scales["hip"]
        )
        # step frequency 
        
        feet_contact_force =  torch.norm(self._robots._feet.get_net_contact_forces(clone=False).view(self._num_envs, 2, 3), dim=-1)
        feet_contact = torch.where(feet_contact_force[:,:] > 0.1, 1, 0)
        feet_contact_state_change = feet_contact  - self.last_feet_touch_state 
        self.last_feet_touch_state = feet_contact

        now_time = self.progress_buf.reshape((self.num_envs,1)).expand(-1, 2)
        self.feet_contact_time = torch.where(feet_contact_state_change == 1,now_time,self.feet_contact_time)
        feet_contact_err = torch.abs(80-torch.abs(torch.clip(self.feet_contact_time[:,0] - self.feet_contact_time[:,1],-80,80)))
        rew_contact_pre = torch.exp(-feet_contact_err/30.0) * self.rew_scales["contact_pre"]

        # total reward
        self.rew_buf = (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_orient
            + rew_base_height
            + rew_torque
            + rew_joint_acc
            + rew_action_rate
            + rew_hip
            + rew_fallen_over
            + rew_contact_pre
        )
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip
        self.episode_sums["step_pre"] += rew_contact_pre

    def get_observations(self):
        # self.measured_heights = self.get_heights()
        # heights = (
        #     torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.height_meas_scale
        # )
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel * self.ang_vel_scale,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_pos_sel * self.dof_pos_scale,
                self.dof_vel_sel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )

    def get_ground_heights_below_knees(self):
        
        points = self.knee_pos.reshape(self.num_envs, 4, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_ground_heights_below_base(self):
        points = self.base_pos.reshape(self.num_envs, 1, 3)
        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, 0:3]
            ).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))
