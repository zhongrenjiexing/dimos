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