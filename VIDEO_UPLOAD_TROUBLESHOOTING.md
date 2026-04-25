# 视频上传模式问题排查指南

## ✅ 已完成的修复

### 1. 后端 API 端点添加
已在 `api_service/main.py` 中添加 `/api/detect-base64` 端点：
- 接收 base64 编码的图片数据
- 解码并转换为 PIL Image
- 调用 pipeline 进行推理
- 返回检测结果

### 2. 前端 API 调用
已在 `web_intf/src/services/api.js` 中添加 `detectEmotionFromBase64` 函数

### 3. 视频帧处理逻辑
已在 `VideoResultsDisplay.jsx` 中实现真实的帧捕获和 API 调用

---

## 🔍 常见问题排查步骤

### 问题 1: 视频上传后没有反应

**检查清单：**
1. ✅ 后端是否正在运行？
   ```bash
   cd api_service
   python main.py
   # 应该看到: INFO: Uvicorn running on http://0.0.0.0:8000
   ```

2. ✅ 前端是否正在运行？
   ```bash
   cd web_intf
   npm run dev
   # 应该看到: Local: http://localhost:5173/
   ```

3. ✅ 浏览器控制台是否有错误？
   - 按 F12 打开开发者工具
   - 查看 Console 标签页
   - 查看 Network 标签页，看是否有失败的 API 请求

### 问题 2: API 连接失败

**可能原因：**
- 后端未启动
- 端口被占用
- CORS 配置问题

**解决方案：**
```bash
# 重启后端
cd api_service
python main.py

# 检查 API 健康状态
curl http://localhost:8000/health
# 应该返回: {"status":"healthy","models_loaded":true,"device":"CPU"}
```

### 问题 3: 视频播放但无检测结果

**可能原因：**
- 视频中没有狗
- 检测置信度阈值太高（当前设置为 0.5）
- 视频质量太差

**调试方法：**
1. 在浏览器控制台查看 API 响应：
   ```javascript
   // 在 Network 标签页中查看 /api/detect-base64 的响应
   ```

2. 检查后端日志：
   ```bash
   # 后端终端会显示推理日志
   ```

3. 尝试使用包含清晰狗脸的视频

### 问题 4: 处理速度太慢

**优化建议：**
- 在 GPU 上运行后端（当前可能在 CPU 上）
- 降低视频分辨率
- 增加 CAPTURE_INTERVAL（当前为 3000ms）

**修改帧捕获间隔：**
```javascript
// 在 VideoResultsDisplay.jsx 第 7 行
const CAPTURE_INTERVAL = 5000; // 改为 5 秒
```

---

## 🧪 测试流程

### 步骤 1: 启动服务
```bash
# Terminal 1 - 后端
cd api_service
python main.py

# Terminal 2 - 前端
cd web_intf
npm run dev
```

### 步骤 2: 访问应用
打开浏览器访问: http://localhost:5173

### 步骤 3: 切换到视频模式
点击右上角 "🎬 Upload Video" 按钮

### 步骤 4: 上传视频
- 拖放或点击上传区域
- 选择包含狗的 MP4 视频文件（< 50MB）

### 步骤 5: 观察行为
预期结果：
1. ✅ 视频自动播放
2. ✅ 每 3 秒显示 "Analyzing frame..." 遮罩
3. ✅ 状态指示器从 "Ready" → "Processing..." → "Analysis Active"
4. ✅ 如果检测到狗，下方显示检测卡片
5. ✅ 检测卡片包含：Dog ID、情绪标签、置信度、概率条

---

## 📊 调试技巧

### 1. 检查 API 是否正常工作
```bash
# 测试图片上传端点
curl -X POST http://localhost:8000/api/detect \
  -F "file=@test_image.jpg"

# 测试 base64 端点（需要先准备 base64 字符串）
curl -X POST http://localhost:8000/api/detect-base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "YOUR_BASE64_STRING"}'
```

### 2. 浏览器控制台日志
在 `VideoResultsDisplay.jsx` 中添加更多日志：
```javascript
console.log('Capturing frame...');
console.log('Base64 length:', base64Image.length);
console.log('API response:', results);
```

### 3. 网络请求检查
- 打开浏览器开发者工具 (F12)
- 切换到 Network 标签
- 过滤 "detect-base64"
- 查看请求和响应的详细信息

---

## 🐛 已知限制

1. **性能**: CPU 模式下处理速度较慢，建议 GPU
2. **格式**: 主要支持 MP4，其他格式可能兼容性不佳
3. **文件大小**: 限制 50MB，大文件可能导致内存问题
4. **实时性**: 每 3 秒处理一帧，不是真正的实时

---

## 💡 常见问题 FAQ

**Q: 为什么视频播放了但没有检测结果？**
A: 可能视频中狗的脸不够清晰，或者角度不好。尝试使用正面清晰的狗脸视频。

**Q: 可以调整检测频率吗？**
A: 可以，修改 `VideoResultsDisplay.jsx` 中的 `CAPTURE_INTERVAL` 常量。

**Q: 支持多长的视频？**
A: 理论上无限制，但建议 < 5 分钟以避免内存问题。

**Q: 可以同时处理多个视频吗？**
A: 当前不支持，一次只能处理一个视频。

---

## 📞 需要帮助？

如果以上步骤都无法解决问题，请提供：
1. 浏览器控制台的错误信息截图
2. Network 标签页中失败的 API 请求详情
3. 后端终端的错误日志
4. 使用的视频文件格式和大小
