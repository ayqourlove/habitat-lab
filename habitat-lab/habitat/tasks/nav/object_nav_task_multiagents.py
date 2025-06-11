import os
from typing import TYPE_CHECKING, Dict, Any, Tuple, Union, Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R

import copy
import attr
import quaternion
import numpy as np
from gym import spaces
import re
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
    HeadingSensor,
)

from habitat.core.embodied_task import (
    Measure,
)

from collections import OrderedDict
from habitat.core.dataset import Dataset, Episode


from habitat.core.simulator import (
    Simulator, Sensor, SensorSuite, SensorTypes, AgentState, RGBSensor, ShortestPathPoint
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.rearrange.utils import (
    UsesArticulatedAgentInterface,
    rearrange_collision,
    rearrange_logger,
)

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_task(name="ObjectNav-multiagents")
class ObjectNavigationMultiAgents(NavigationTask):
    r"""Multi-agent Object Navigation Task class for a task specific methods."""

    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Modifies the sensor suite in place to duplicate articulated agent specific sensors
        between the two articulated agents.
        """

        task_new_sensors: Dict[str, Sensor] = {}
        task_obs_spaces = OrderedDict()
        for agent_idx, agent_id in enumerate(self._sim.agent_names):
            for sensor_name, sensor in sensor_suite.sensors.items():
                new_sensor = copy.copy(sensor)
                new_sensor.agent_id = agent_idx
                full_name = f"{agent_id}_{sensor_name}"
                task_new_sensors[full_name] = new_sensor
                task_obs_spaces[full_name] = new_sensor.observation_space

        sensor_suite.sensors = task_new_sensors
        sensor_suite.observation_spaces = spaces.Dict(spaces=task_obs_spaces)


    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        # self._duplicate_sensor_suite(self.sensor_suite)

    def get_n_yaw_rotations(self, rotation0:list, n:int) -> List[List[float]]:
        """ gpt generated code
        给定一个初始 rotation0四元数和整数 n
        生成绕 y 轴旋转等分的 n 个四元数。
        返回值是一个列表,每个元素是一个长度为 4 的list。
        """
        r0 = R.from_quat(rotation0)  # 初始四元数
        step_angle = 360.0 / n       # 每步角度（度）

        rotations = []
        for i in range(n):
            # 绕Y轴的旋转四元数，注意是 degrees
            delta = R.from_euler("y", i * step_angle, degrees=True)
            r = delta * r0  # 四元数乘法：先转 delta，再转 r0（绕Y轴后套初始姿态）
            rotations.append(r.as_quat().tolist())  # 转回 list 格式 [x, y, z, w]

        return rotations    

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                # Set the start position and rotation for each agent based on the episode
                # Every agent will have the same start position but different rotations
                rotations = self.get_n_yaw_rotations(
                    episode.start_rotation, len(config.simulator.agents))
                for agent_id in range(len(config.simulator.agents)):
                    agent_config = get_agent_config(config.simulator, agent_id=agent_id)
                    agent_config.start_position = episode.start_position
                    agent_config.start_rotation = [
                        float(k) for k in rotations[agent_id]
                    ]
                    agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)

    def reset(self, episode: Episode):
        observations = self._sim.reset()
        for agent_id in self._sim.agent_ids:
            temp = self.sensor_suite.get_observations(
                observations=observations[agent_id],
                episode=episode,
                task=self,
                should_time=True,
                agent_id=agent_id,
            )
            observations[agent_id].update(temp)

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        self._is_episode_active = True

        return observations

    def step(self, action: Dict[int, Dict[str, Any]], episode: Episode):
        is_done = False
        for a in action.values():
            if "action_args" not in a or a["action_args"] is None:
                a["action_args"] = {}
        observations: Optional[Any] = None
        for a in action.values():
            action_name = a["action"]
            if action_name == 0:
                is_done = True
                agent_action_done = a
                break
        if is_done:
            task_action = self.actions["stop"]
            task_action.step(**agent_action_done["action_args"], task=self)
            observations = self._sim.get_sensor_observations(
                agent_ids=self._sim.agent_ids)
        else:
            sim_action = {k: v["action"] for k, v in action.items()}
            observations = self._sim.step(sim_action)

        self._sim.step_physics(1.0 / self._physics_target_sps)  # type:ignore

        if observations is None:
            observations = self._sim.step(None)
        
        for agent_id in self._sim.agent_ids:
            temp = self.sensor_suite.get_observations(
                observations=observations[agent_id],
                episode=episode,
                task=self,
                should_time=True,
                agent_id=agent_id,
            )
            observations[agent_id].update(temp)

        self._is_episode_active = all(self._check_episode_is_active(
            observations=observations[agent_id], action=action, episode=episode
        ) for agent_id in self._sim.agent_ids)

            
        return observations


@registry.register_sensor(name="CompassSensor_multiagents")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the episode,
    theta=0 is defined by the agents state at t=0
    multiagents version
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, agent_id, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state(agent_id=agent_id)
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="GPSSensor_multiagents")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0
    multiagents version
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the `dimensionality` field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, agent_id, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state(agent_id=agent_id)

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)

@registry.register_measure
class DistanceToGoal_multiagents(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._distance_to = self._config.distance_to

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = [None for _ in self._sim.agent_ids]
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        distance_to_target: list = []
        for agent_id in self._sim.agent_ids:
            current_position = self._sim.get_agent_state(agent_id=agent_id).position

            if self._previous_position[agent_id] is None or not np.allclose(
                self._previous_position[agent_id], current_position, atol=1e-4
            ):
                if self._distance_to == "POINT":
                    distance_to_target.append(self._sim.geodesic_distance(
                        current_position,
                        [goal.position for goal in episode.goals],
                        episode,
                    ))
                elif self._distance_to == "VIEW_POINTS":
                    distance_to_target.append(self._sim.geodesic_distance(
                        current_position, self._episode_view_points, episode
                    ))
                else:
                    logger.error(
                        f"Non valid distance_to parameter was provided: {self._distance_to }"
                    )

                self._previous_position[agent_id] = (
                    current_position[0],
                    current_position[1],
                    current_position[2],
                )
                self._metric = min(distance_to_target)