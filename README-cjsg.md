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

yolo v11 提取目标，投影到3d点云上
```bash  
dimos --replay --replay-dir unitree_go2_office_walk2 run unitree-go2-detection
```
