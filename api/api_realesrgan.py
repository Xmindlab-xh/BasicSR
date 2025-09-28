import os
import shutil
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


def get_unique_name():
    return str(uuid.uuid4())


def run_realesrgan_image(
    input_path: str, output_path: str, model_name="RealESRGAN_x4plus", suffix=""
):
    cmd = [
        "python",
        "inference/inference_realesrgan.py",
        "-i",
        input_path,
        "-o",
        output_path,
        "-n",
        model_name,
        "--suffix",
        suffix,
    ]
    subprocess.run(cmd, check=True)


def run_realesrgan_video(
    input_path: str, output_path: str, model_name="realesr-animevideov3", suffix=""
):
    # 避免并行进程数为0导致报错
    num_process = 1
    cmd = [
        "python",
        "inference/inference_realesrgan_video.py",
        "-i",
        input_path,
        "-o",
        output_path,
        "-n",
        model_name,
        "--suffix",
        suffix,
        "--num_process",
        str(num_process),
    ]
    subprocess.run(cmd, check=True)


@app.post("/superres-image")
async def superres_image(file: UploadFile = File(...)):
    file_id = get_unique_name()
    input_path = os.path.join(TMP_DIR, f"{file_id}_input.jpg")
    output_path = os.path.join(TMP_DIR, file_id)  # 目录形式，避免 suffix 冲突

    # 保存上传文件
    with open(input_path, "wb") as f:
        f.write(await file.read())

    os.makedirs(output_path, exist_ok=True)

    # 调用 RealESRGAN
    run_realesrgan_image(input_path, output_path, suffix=file_id)

    # 拼接输出文件路径
    result_file = os.path.join(
        output_path,
        f"{os.path.splitext(os.path.basename(input_path))[0]}_{file_id}.png",
    )
    if not os.path.exists(result_file):
        raise RuntimeError(f"输出文件不存在: {result_file}")

    return FileResponse(result_file, filename=f"sr_{file.filename}")


@app.post("/superres-video")
async def superres_video(file: UploadFile = File(...)):
    file_id = get_unique_name()
    input_path = os.path.join(TMP_DIR, f"{file_id}_input.mp4")
    output_path = os.path.join(TMP_DIR, file_id)  # 目录形式

    # 保存上传文件
    with open(input_path, "wb") as f:
        f.write(await file.read())

    os.makedirs(output_path, exist_ok=True)

    # 调用 RealESRGAN 视频处理
    run_realesrgan_video(input_path, output_path, suffix=file_id)

    # 拼接输出文件路径
    # 视频输出通常是 output_path 下的 {原文件名}_{suffix}.mp4
    result_file = os.path.join(
        output_path,
        f"{os.path.splitext(os.path.basename(input_path))[0]}_{file_id}.mp4",
    )
    if not os.path.exists(result_file):
        raise RuntimeError(f"输出文件不存在: {result_file}")

    return FileResponse(result_file, filename=f"sr_{file.filename}")
