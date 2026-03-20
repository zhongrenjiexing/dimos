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


from typing import TYPE_CHECKING, Any

import json
import os
from dataclasses import dataclass
from pathlib import Path
import threading

from dimos_lcm.foxglove_msgs.ImageAnnotations import (
    ImageAnnotations,
)
from lcm_msgs.foxglove_msgs import SceneUpdate  # type: ignore[import-not-found]
from reactivex import operators as ops
from reactivex.observable import Observable

from dimos import spec
from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module_coordinator import ModuleCoordinator
from dimos.core.stream import In, Out
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Transform, Vector3
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.perception.detection.module2D import Detection2DModule, Config as Detection2DConfig
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection3d import Detection3DPC
from dimos.perception.detection.type.detection3d.imageDetections3DPC import ImageDetections3DPC
from dimos.types.timestamped import align_timestamped
from dimos.utils.reactive import backpressure
from dimos.utils.logging_config import get_run_log_dir, setup_logger

logger = setup_logger()

if TYPE_CHECKING:
    from dimos.core.rpc_client import ModuleProxy


@dataclass
class Detection3DConfig(Detection2DConfig):
    # Save per-detection 3D positions as JSONL (one object per line).
    # Uses the current DimOS run log directory if available.
    save_object_positions: bool = False
    object_positions_filename: str = "object_positions_{pid}.jsonl"
    include_bbox_dimensions: bool = True


class Detection3DModule(Detection2DModule):
    default_config = Detection3DConfig

    color_image: In[Image]
    pointcloud: In[PointCloud2]

    detections: Out[Detection2DArray]
    annotations: Out[ImageAnnotations]
    scene_update: Out[SceneUpdate]

    # just for visualization,
    # emits latest pointclouds of detected objects in a frame
    detected_pointcloud_0: Out[PointCloud2]
    detected_pointcloud_1: Out[PointCloud2]
    detected_pointcloud_2: Out[PointCloud2]

    # just for visualization, emits latest top 3 detections in a frame
    detected_image_0: Out[Image]
    detected_image_1: Out[Image]
    detected_image_2: Out[Image]

    detection_3d_stream: Observable[ImageDetections3DPC] | None = None

    _object_positions_lock: threading.RLock
    _object_positions_fh: Any | None = None
    _object_positions_path: Path | None = None

    def process_frame(
        self,
        detections: ImageDetections2D,
        pointcloud: PointCloud2,
        transform: Transform,
    ) -> ImageDetections3DPC:
        if not transform:
            return ImageDetections3DPC(detections.image, [])

        detection3d_list: list[Detection3DPC] = []
        for detection in detections:
            detection3d = Detection3DPC.from_2d(
                detection,
                world_pointcloud=pointcloud,
                camera_info=self.config.camera_info,
                world_to_optical_transform=transform,
            )
            if detection3d is not None:
                detection3d_list.append(detection3d)

        return ImageDetections3DPC(detections.image, detection3d_list)

    def pixel_to_3d(
        self,
        pixel: tuple[int, int],
        assumed_depth: float = 1.0,
    ) -> Vector3:
        """Unproject 2D pixel coordinates to 3D position in camera optical frame.

        Args:
            camera_info: Camera calibration information
            assumed_depth: Assumed depth in meters (default 1.0m from camera)

        Returns:
            Vector3 position in camera optical frame coordinates
        """
        # Extract camera intrinsics
        fx, fy = self.config.camera_info.K[0], self.config.camera_info.K[4]
        cx, cy = self.config.camera_info.K[2], self.config.camera_info.K[5]

        # Unproject pixel to normalized camera coordinates
        x_norm = (pixel[0] - cx) / fx
        y_norm = (pixel[1] - cy) / fy

        # Create 3D point at assumed depth in camera optical frame
        # Camera optical frame: X right, Y down, Z forward
        return Vector3(x_norm * assumed_depth, y_norm * assumed_depth, assumed_depth)

    @skill
    def ask_vlm(self, question: str) -> str:
        """asks a visual model about the view of the robot, for example
        is the bannana in the trunk?
        """
        from dimos.models.vl.qwen import QwenVlModel

        model = QwenVlModel()
        image = self.color_image.get_next()
        return model.query(image, question)

    # @skill
    @rpc
    def nav_vlm(self, question: str) -> str:
        """
        query visual model about the view in front of the camera
        you can ask to mark objects like:

        "red cup on the table left of the pencil"
        "laptop on the desk"
        "a person wearing a red shirt"
        """
        from dimos.models.vl.qwen import QwenVlModel

        model = QwenVlModel()
        image = self.color_image.get_next()
        result = model.query_detections(image, question)

        print("VLM result:", result, "for", image, "and question", question)

        if isinstance(result, str) or not result or not len(result):
            return None  # type: ignore[return-value]

        detections: ImageDetections2D = result

        print(detections)
        if not len(detections):
            print("No 2d detections")
            return None  # type: ignore[return-value]

        pc = self.pointcloud.get_next()
        transform = self.tf.get("camera_optical", pc.frame_id, detections.image.ts, 5.0)

        detections3d = self.process_frame(detections, pc, transform)

        if len(detections3d):
            return detections3d[0].pose  # type: ignore[no-any-return]
        print("No 3d detections, projecting 2d")

        center = detections[0].get_bbox_center()
        return PoseStamped(
            ts=detections.image.ts,
            frame_id="world",
            position=self.pixel_to_3d(center, assumed_depth=1.5),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )

    @rpc
    def start(self) -> None:
        super().start()

        self._object_positions_lock = threading.RLock()
        self._object_positions_fh = None
        self._object_positions_path = None

        if self.config.save_object_positions:
            run_log_dir = get_run_log_dir()
            base_dir = run_log_dir if run_log_dir is not None else Path.cwd()

            filename = self.config.object_positions_filename
            pid = os.getpid()
            filename = filename.format(pid=pid)

            out_path = Path(base_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # buffering=1 enables line-buffered writes for JSONL.
            self._object_positions_fh = open(out_path, "a", encoding="utf-8", buffering=1)
            self._object_positions_path = out_path
            logger.info(f"[Detection3D] saving object positions to {out_path}")

        def detection2d_to_3d(args):  # type: ignore[no-untyped-def]
            detections, pc = args
            transform = self.tf.get("camera_optical", pc.frame_id, detections.image.ts, 5.0)
            return self.process_frame(detections, pc, transform)

        self.detection_stream_3d = align_timestamped(
            backpressure(self.detection_stream_2d()),
            self.pointcloud.observable(),  # type: ignore[no-untyped-call]
            match_tolerance=0.25,
            buffer_size=20.0,
        ).pipe(ops.map(detection2d_to_3d))

        self.detection_stream_3d.subscribe(self._publish_detections)

    @rpc
    def stop(self) -> None:
        # Close append-only file handle to ensure the last buffered JSON line is flushed.
        if getattr(self, "_object_positions_fh", None) is not None:
            with self._object_positions_lock:
                try:
                    self._object_positions_fh.close()  # type: ignore[union-attr]
                finally:
                    self._object_positions_fh = None

        super().stop()

    def _publish_detections(self, detections: ImageDetections3DPC) -> None:
        if not detections:
            return

        for det in detections:
            center = det.center
            w, h, d = det.get_bounding_box_dimensions()
            n_pts = len(det.pointcloud.pointcloud.points)
            print(
                f"[Detection3D] {det.name:15s} | id={det.track_id}"
                f" | conf={det.confidence:.0%}"
                f" | center=({center.x:.2f}, {center.y:.2f}, {center.z:.2f})"
                f" | size=({w:.2f}x{h:.2f}x{d:.2f})m"
                f" | pts={n_pts}"
            )

            if self._object_positions_fh is not None:
                record: dict[str, Any] = {
                    "ts": det.ts,
                    "frame_id": det.frame_id,
                    "label": det.name,
                    "track_id": det.track_id,
                    "confidence": det.confidence,
                    "x": center.x,
                    "y": center.y,
                    "z": center.z,
                    "num_points": n_pts,
                }
                if self.config.include_bbox_dimensions:
                    record["bbox_dimensions_m"] = {"w": w, "h": h, "d": d}

                # One JSON record per line (append-only).
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with self._object_positions_lock:
                    self._object_positions_fh.write(line)

        for index, detection in enumerate(detections[:3]):
            pointcloud_topic = getattr(self, "detected_pointcloud_" + str(index))
            pointcloud_topic.publish(detection.pointcloud)

        self.scene_update.publish(detections.to_foxglove_scene_update())


def deploy(  # type: ignore[no-untyped-def]
    dimos: ModuleCoordinator,
    lidar: spec.Pointcloud,
    camera: spec.Camera,
    prefix: str = "/detector3d",
    **kwargs,
) -> "ModuleProxy":
    detector = dimos.deploy(Detection3DModule, camera_info=camera.hardware_camera_info, **kwargs)  # type: ignore[attr-defined]

    detector.image.connect(camera.color_image)
    detector.pointcloud.connect(lidar.pointcloud)

    detector.annotations.transport = LCMTransport(f"{prefix}/annotations", ImageAnnotations)
    detector.detections.transport = LCMTransport(f"{prefix}/detections", Detection2DArray)
    detector.scene_update.transport = LCMTransport(f"{prefix}/scene_update", SceneUpdate)

    detector.detected_image_0.transport = LCMTransport(f"{prefix}/image/0", Image)
    detector.detected_image_1.transport = LCMTransport(f"{prefix}/image/1", Image)
    detector.detected_image_2.transport = LCMTransport(f"{prefix}/image/2", Image)

    detector.detected_pointcloud_0.transport = LCMTransport(f"{prefix}/pointcloud/0", PointCloud2)
    detector.detected_pointcloud_1.transport = LCMTransport(f"{prefix}/pointcloud/1", PointCloud2)
    detector.detected_pointcloud_2.transport = LCMTransport(f"{prefix}/pointcloud/2", PointCloud2)

    detector.start()

    return detector


detection3d_module = Detection3DModule.blueprint

__all__ = ["Detection3DModule", "deploy", "detection3d_module"]
