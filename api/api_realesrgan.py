from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import subprocess
import sys

app = FastAPI(title="Real-ESRGAN API with Auto Cleanup")

# 临时目录
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 项目根目录和 inference 路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")


def cleanup_files(*file_paths):
    """删除临时文件"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"清理文件失败: {path}, 错误: {e}")


# 图片超分接口
@app.post("/superres-image")
async def superres_image(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    file_ext = os.path.splitext(file.filename)[1]
    img_id = str(uuid.uuid4())

    # 输入输出路径
    input_path = os.path.join(TMP_DIR, f"{img_id}{file_ext}")
    output_path = os.path.join(TMP_DIR, f"{img_id}_out{file_ext}")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用 inference 脚本
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
        "out",
    ]
    subprocess.run(cmd, check=True)

    # 后台清理
    background_tasks.add_task(cleanup_files, input_path, output_path)

    return FileResponse(
        output_path, filename=f"superres_{file.filename}", background=background_tasks
    )


# 视频超分接口
@app.post("/superres-video")
async def superres_video(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    file_ext = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())

    # 输入输出路径
    input_path = os.path.join(TMP_DIR, f"{video_id}{file_ext}")
    output_path = os.path.join(TMP_DIR, f"{video_id}_out.mp4")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用 inference 脚本
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
        "out",
    ]
    subprocess.run(cmd, check=True)

    # 后台清理
    background_tasks.add_task(cleanup_files, input_path, output_path)

    return FileResponse(
        output_path, filename=f"superres_{file.filename}", background=background_tasks
    )
