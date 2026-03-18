## (连接手机热点下的PC操作-打通 手机热点192.168.43.xx与go2w 192.168.123.xx)：
1.
第一步：在thinkpad-p16终端添加路由
```bash
sudo ip route add 192.168.123.0/24 via 192.168.43.199
```
2
第二步：验证可以 ping 通运动控制器
```bash
ping -c 3 192.168.123.161
```
3
第三步：测试 WebRTC 信令端口
```bash
curl -s http://192.168.123.161:9991/con_notify | base64 -d | python3 -m json.tool
```
4
第四步：用正确的 IP 运行 dimos
```bash
export ROBOT_IP=192.168.123.161
cd ~/dimos/dimos
dimos run unitree-go2
```


服务器上启动的localhost:8011/v1
```bash
export OPENAI_BASE_URL=https://nat-notebook-inspire.sii.edu.cn/ws-6040202d-b785-4b37-98b0-c68d65dd52ce/project-4493c9f7-2fbf-459a-ad90-749a5a420b91/user-df257773-23f8-4056-a239-ad9fda140fa1/vscode/ec733ace-a9cb-494d-929b-71c7748a10b1/11653f77-e6f6-420f-91cc-a80ef22a9758/proxy/8011/v1
```

```bash
export OPENAI_BASE_URL=http://<服务器IP>:8011/v1
export OPENAI_API_KEY=any-key
export DIMOS_VLM_MODEL=qwen2.5-vl
```

## README 中的 Spatial Memory / Object 相关 demo

### 1. 物体检测 + 3D 坐标 + Rerun 可视化

| 功能 | 命令 | 说明 |
|------|------|------|
| **Object localization** | `dimos run demo-object-scene-registration` | 需要 ZED/RealSense 深度相机，用 YOLO-E 检测物体 + 3D 投影，输出到 Foxglove |
| **Spatio-temporal RAG** | `dimos run unitree-go2-temporal-memory` | VLM 分析视频帧，提取实体，在 Rerun 中绘制 3D 物体标记点 |
| **Spatial Memory** | `dimos run unitree-go2-spatial` | CLIP 索引场景，语义检索 |

### 2. 物体检测与 Rerun 绘制代码位置

- **物体检测**：`dimos/perception/object_scene_registration.py` — `ObjectSceneRegistrationModule`（YOLO-E 2D → 深度 3D 投影）
- **3D 标记点绘制**：`dimos/msgs/visualization_msgs/EntityMarkers.py` — `to_rerun()` 转为 `rr.Points3D`
- **点云/体素**：`dimos/msgs/sensor_msgs/PointCloud2.py` — `to_rerun()` 转为 Points3D/Boxes3D
- **Rerun 桥接**：`dimos/visualization/rerun/bridge.py` — `RerunBridgeModule` 自动订阅 LCM 并调用 `to_rerun()`

### 3. onnxruntime 调用位置

- `dimos/agents_deprecated/memory/image_embedding.py` — CLIP 图像嵌入（SpatialMemory 用）
- `dimos/simulation/mujoco/policy.py` — 策略推理


