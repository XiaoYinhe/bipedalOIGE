
import math

import numpy as np
import torch
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.articulations.motorTest import MotorTest

from omniisaacgymenvs.tasks.utils.anymal_terrain_generator import *
from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from pxr import UsdLux, UsdPhysics

import csv


class MotorTestTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        
        self.update_config(sim_config)
        

        self._num_observations = 2
        self._num_actions = 1

        RLTask.__init__(self, name, env)

        # self.height_points = self.init_height_points()
        
        # 写入CSV文件
        csvfile = open("isaac_pd_20_2.csv", 'w', newline='', encoding='utf-8') 
        self.csv_writer = csv.writer(csvfile)
        column_name = ["time","setPos","fbPos"]
        # 写入表头（如果有多个列的话，这里会是多个列名）
        self.csv_writer.writerow(column_name)
        self.loaclTime= torch.zeros((1,1),device = self._device)
        self.setPos = torch.ones((1,1),device= self._device) * 3
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])



    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_robot()
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])
        
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/Robot", name="robot_view",reset_xform_properties=False
        )
        scene.add(self._robots)


    def get_robot(self):
        init_translation = torch.tensor([0.0, 0.0, 0])
        init_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        robot = MotorTest(
            prim_path=self.default_zero_env_path + "/Robot",
            name="Robot",
            translation=init_translation,
            orientation=init_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "bipedal", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("bipedal")
        )


    def get_observations(self) -> dict:
        observations = {self._robots.name: {"obs_buf": torch.zeros((self.num_envs,2),device=self._device)}}
        return observations
    
    def post_reset(self):
       
        # initialize some data used later on
        self.up_axis_idx = 2

        self.num_dof = self._robots.num_dof
        
    def reset_idx(self, env_ids):
        return

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return

        for i in range(4):
            if self.world.is_playing():
                
                dof_pos = self._robots.get_joint_positions(clone=False)
                dof_vel = self._robots.get_joint_velocities(clone=False)
                self.loaclTime += 0.005
                data_list = torch.cat(
                    (
                        self.loaclTime,
                        self.setPos,
                        dof_pos,
                    ),
                    dim=-1,
                )
            
                self.csv_writer.writerow(*data_list.tolist())

                torques = torch.clip(
                    20 * (3 - dof_pos)
                    - 2 * dof_vel,
                    -40,
                    40,
                )
                
                self._robots.set_joint_efforts(torques)
                SimulationContext.step(self.world, render=False)
                

    def post_physics_step(self):
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def calculate_metrics(self):
        self.rew_buf = torch.tensor([[0]],device=self._device);

