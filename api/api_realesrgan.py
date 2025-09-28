from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import subprocess

app = FastAPI(title="Real-ESRGAN API")

# 输出临时目录
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")


# 视频处理接口
@app.post("/superres-video")
async def superres_video(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    file_ext = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{video_id}_input{file_ext}")
    output_path = os.path.join(TMP_DIR, f"{video_id}_output.mp4")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 后台处理任务
    def process_video():
        # 调用 inference_realesrgan_video.py
        cmd = [
            "python",
            os.path.join(INFERENCE_DIR, "inference_realesrgan.py"),
            "-i",
            input_path,
            "-o",
            TMP_DIR,
            "-n",
            "realesr-animevideov3",
            "--suffix",
            video_id,
        ]
        subprocess.run(cmd, check=True)

    background_tasks.add_task(process_video)

    return {"message": "Video is being processed", "output_path": output_path}


# 图片处理接口
@app.post("/superres-image")
async def superres_image(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = None
):
    file_ext = os.path.splitext(file.filename)[1]
    img_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{img_id}_input{file_ext}")
    output_path = os.path.join(TMP_DIR, f"{img_id}_output{file_ext}")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 后台处理任务
    def process_image():
        # 调用 inference_realesrgan.py
        cmd = [
            "python",
            os.path.join(INFERENCE_DIR, "inference_realesrgan_video.py"),
            "-i",
            input_path,
            "-o",
            TMP_DIR,
            "-n",
            "RealESRGAN_x4plus",
            "--suffix",
            img_id,
        ]
        subprocess.run(cmd, check=True)

    background_tasks.add_task(process_image)

    return {"message": "Image is being processed", "output_path": output_path}


# 文件下载接口
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}
