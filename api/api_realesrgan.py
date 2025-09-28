from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import subprocess
import sys
import glob
import time

app = FastAPI(title="Real-ESRGAN API with Auto Cleanup")

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")


def cleanup_files(*file_paths):
    """后台删除临时文件"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"清理文件失败: {path}, 错误: {e}")


def find_latest_file(pattern: str, timeout=60):
    """
    根据通配符查找最新的文件，等待最多timeout秒
    """
    start = time.time()
    while time.time() - start < timeout:
        files = glob.glob(pattern)
        if files:
            # 返回最新修改的文件
            return max(files, key=os.path.getmtime)
        time.sleep(0.5)
    return None


@app.post("/superres-image")
async def superres_image(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    ext = os.path.splitext(file.filename)[1]
    img_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{img_id}_input{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [
        sys.executable,
        os.path.join(INFERENCE_DIR, "inference_realesrgan.py"),
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

    # 动态查找输出
    pattern = os.path.join(TMP_DIR, f"*{img_id}*{ext}")
    result_file = find_latest_file(pattern)

    if not result_file or not os.path.exists(result_file):
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail="处理图片失败，未生成输出文件")

    background_tasks.add_task(cleanup_files, input_path, result_file)

    return FileResponse(
        result_file, filename=f"superres_{file.filename}", background=background_tasks
    )


@app.post("/superres-video")
async def superres_video(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    ext = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{video_id}_input{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cmd = [
        sys.executable,
        os.path.join(INFERENCE_DIR, "inference_realesrgan_video.py"),
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

    # 动态查找输出
    pattern = os.path.join(TMP_DIR, f"*{video_id}*.mp4")
    result_file = find_latest_file(pattern)

    if not result_file or not os.path.exists(result_file):
        cleanup_files(input_path)
        raise HTTPException(status_code=500, detail="处理视频失败，未生成输出文件")

    background_tasks.add_task(cleanup_files, input_path, result_file)

    return FileResponse(
        result_file, filename=f"superres_{file.filename}", background=background_tasks
    )
