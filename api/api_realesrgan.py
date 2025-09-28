from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import subprocess
import sys

app = FastAPI(title="Real-ESRGAN API with Auto Cleanup")

# 输出临时目录
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# 绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")


def cleanup_files(*file_paths):
    """删除临时文件"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"清理文件失败: {path}, 错误: {e}")


# 图片处理接口
@app.post("/superres-image")
async def superres_image(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    file_ext = os.path.splitext(file.filename)[1]
    img_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{img_id}_input{file_ext}")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用 inference 脚本处理图片
    cmd = [
        sys.executable,  # 确保用 venv 里的 Python
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

    # inference 会生成 inputname_{suffix}.ext
    input_basename, _ = os.path.splitext(os.path.basename(input_path))
    result_file = os.path.join(TMP_DIR, f"{input_basename}_{img_id}{file_ext}")

    # 设置后台任务删除临时文件
    background_tasks.add_task(cleanup_files, input_path, result_file)

    # 返回文件
    return FileResponse(
        result_file, filename=f"superres_{file.filename}", background=background_tasks
    )


# 视频处理接口
@app.post("/superres-video")
async def superres_video(
    file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()
):
    file_ext = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{video_id}_input{file_ext}")

    # 保存上传文件
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 调用 inference 脚本处理视频
    cmd = [
        sys.executable,
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

    # inference 会生成 inputname_{suffix}.mp4
    input_basename, _ = os.path.splitext(os.path.basename(input_path))
    result_file = os.path.join(TMP_DIR, f"{input_basename}_{video_id}.mp4")

    # 设置后台任务删除临时文件
    background_tasks.add_task(cleanup_files, input_path, result_file)

    # 返回文件
    return FileResponse(
        result_file, filename=f"superres_{file.filename}", background=background_tasks
    )
