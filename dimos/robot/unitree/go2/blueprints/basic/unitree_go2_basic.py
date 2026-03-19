#!/usr/bin/env python3

# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
from typing import Any

from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import pSHMTransport
from dimos.msgs.sensor_msgs import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.protocol.pubsub.patterns import Glob
from dimos.protocol.service.system_configurator import ClockSyncConfigurator
from dimos.robot.unitree.go2.connection import go2_connection
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

# Mac has some issue with high bandwidth UDP, so we use pSHMTransport for color_image
# actually we can use pSHMTransport for all platforms, and for all streams
# TODO need a global transport toggle on blueprints/global config
_mac_transports: dict[tuple[str, type], pSHMTransport[Image]] = {
    ("color_image", Image): pSHMTransport(
        "color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
}

_transports_base = (
    autoconnect() if platform.system() == "Linux" else autoconnect().transports(_mac_transports)
)


def _convert_camera_info(camera_info: Any) -> Any:
    return camera_info.to_rerun(
        image_topic="/world/color_image",
        optical_frame="camera_optical",
    )


def _convert_global_map(grid: Any) -> Any:
    return grid.to_rerun(voxel_size=0.1, mode="boxes")


def _convert_navigation_costmap(grid: Any) -> Any:
    return grid.to_rerun(
        colormap="Accent",
        z_offset=0.015,
        opacity=0.2,
        background="#484981",
    )


def _static_base_link(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.35, 0.155, 0.2],
            colors=[(0, 255, 127, 60)],
            fill_mode="solid",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _convert_scene_update(scene_update: Any) -> Any:
    """Convert Foxglove SceneUpdate (cube entities) to rr.Boxes3D for Rerun."""
    import numpy as np
    import rerun as rr

    centers = []
    half_sizes = []
    labels = []
    colors = []

    for entity in scene_update.entities:
        for cube in entity.cubes:
            p = cube.pose.position
            s = cube.size
            c = cube.color
            centers.append([p.x, p.y, p.z])
            half_sizes.append([s.x / 2.0, s.y / 2.0, s.z / 2.0])
            # Use the full text label (e.g. "3/person (87%)") if available
            label = entity.texts[0].text if entity.texts_length > 0 else entity.id
            labels.append(label)
            # Force alpha to 200/255 (~78%) so boxes are clearly visible regardless
            # of the original cube.color.a=0.2 set in SceneEntity construction
            colors.append([
                int(c.r * 255),
                int(c.g * 255),
                int(c.b * 255),
                200,
            ])

    if not centers:
        return None

    return rr.Boxes3D(
        centers=np.array(centers, dtype=np.float32),
        half_sizes=np.array(half_sizes, dtype=np.float32),
        labels=labels,
        colors=colors,
        radii=0.02,  # 2cm thick edges so the box outline is clearly visible
    )


def _go2_rerun_blueprint() -> Any:
    """Split layout: camera feed + 3D world view side by side."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Camera"),
            rrb.Spatial3DView(origin="world", name="3D"),
            column_shares=[1, 2],
        ),
    )


rerun_config = {
    "blueprint": _go2_rerun_blueprint,
    # any pubsub that supports subscribe_all and topic that supports str(topic)
    # is acceptable here
    "pubsubs": [LCM()],
    # Custom converters for specific rerun entity paths
    # Normally all these would be specified in their respectative modules
    # Until this is implemented we have central overrides here
    #
    # This is unsustainable once we move to multi robot etc
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/global_map": _convert_global_map,
        "world/navigation_costmap": _convert_navigation_costmap,
        # SceneUpdate (Foxglove format) → rr.Boxes3D, matches any topic ending in /scene_update
        Glob("**/scene_update"): _convert_scene_update,
    },
    # slapping a go2 shaped box on top of tf/base_link
    "static": {
        "world/tf/base_link": _static_base_link,
    },
}


if global_config.viewer == "foxglove":
    from dimos.robot.foxglove_bridge import foxglove_bridge

    with_vis = autoconnect(
        _transports_base,
        foxglove_bridge(shm_channels=["/color_image#sensor_msgs.Image"]),
    )
elif global_config.viewer.startswith("rerun"):
    from dimos.visualization.rerun.bridge import _resolve_viewer_mode, rerun_bridge

    with_vis = autoconnect(
        _transports_base, rerun_bridge(viewer_mode=_resolve_viewer_mode(), **rerun_config)
    )
else:
    with_vis = _transports_base

unitree_go2_basic = (
    autoconnect(
        with_vis,
        go2_connection(),
        websocket_vis(),
    )
    .global_config(n_workers=4, robot_model="unitree_go2")
    .configurators(ClockSyncConfigurator())
)

__all__ = [
    "unitree_go2_basic",
]
