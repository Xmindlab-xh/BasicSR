### 📦 模型说明
1. `**RealESRGAN_x4plus.pth**`
    - **场景**：通用的自然图像（风景、人像、物体）
    - **特点**：4 倍超分，精度高，效果均衡
    - **推荐用途**：普通照片、高清化常规图像
2. `**RealESRNet_x4plus.pth**`
    - **场景**：作为 ESRGAN 的降噪网络
    - **特点**：保守增强，细节更少伪影
    - **推荐用途**：图像噪点很多时，可以先用它清理，再用 `RealESRGAN_x4plus` 做二次放大
3. `**RealESRGAN_x2plus.pth**`
    - **场景**：通用图像，但只需要 **2 倍放大**
    - **特点**：速度比 x4 快，适合显存小、放大倍率需求不高的场景
    - **推荐用途**：低分辨率但不需要放特别大，比如 720p→1080p
4. `**RealESRGAN_x4plus_anime_6B.pth**`
    - **场景**：二次元/动漫插画
    - **特点**：对线条、色块处理更干净，没有过度锐化
    - **推荐用途**：动画截图、漫画、插画、游戏立绘
5. `**realesr-general-x4v3.pth**`
    - **场景**：通用图像，V3 改进版
    - **特点**：更适合多样化输入（照片、截图、UI 等），比 `x4plus` 更灵活
    - **推荐用途**：既有人像、也有文字/截图混合的场景
6. `**realesr-general-wdn-x4v3.pth**`
    - **场景**：带 **降噪 (WDN, with denoise)** 的通用模型
    - **特点**：对噪点强的图像更友好，但会牺牲一些细节
    - **推荐用途**：压缩严重的图（例如 JPG artifacts 很重的），或扫描件
7. `**realesr-animevideov3.pth**`
    - **场景**：动漫视频
    - **特点**：比 `anime_6B` 更适合 **逐帧视频**，在连贯性和速度上更好
    - **推荐用途**：二次元视频超分，比如 B 站、旧番、低清动画

---

### 🔑 使用建议
+ **普通照片** → `RealESRGAN_x4plus.pth`
+ **动漫/插画** → `RealESRGAN_x4plus_anime_6B.pth`
+ **老旧、压缩图像** → `realesr-general-wdn-x4v3.pth`
+ **混合内容（人像+文字+截图）** → `realesr-general-x4v3.pth`
+ **视频动漫** → `realesr-animevideov3.pth`
+ **轻量级 2x 放大** → `RealESRGAN_x2plus.pth`

