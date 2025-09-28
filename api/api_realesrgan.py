import os
import shutil
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()

TMP_DIR = "temp"
os.makedirs(TMP_DIR, exist_ok=True)


def get_unique_name():
    return str(uuid.uuid4())


def run_realesrgan_image(input_path: str, output_path: str, model_name="RealESRGAN_x4plus", suffix=""):
    cmd = [
        "python",
        "inference/inference_realesrgan.py",
        "-i", input_path,
        "-o", output_path,
        "-n", model_name,
        "--suffix", suffix
    ]
    subprocess.run(cmd, check=True)


def run_realesrgan_video(input_path: str, output_path: str, model_name="realesr-animevideov3", suffix=""):
    num_process = 1
    cmd = [
        "python",
        "inference/inference_realesrgan_video.py",
        "-i", input_path,
        "-o", output_path,
        "-n", model_name,
        "--suffix", suffix,
        "--num_process", str(num_process)
    ]
    subprocess.run(cmd, check=True)


@app.post("/superres-image")
async def superres_image(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    filename_base = f"{os.path.splitext(file.filename)[0]}_{unique_id}"
    input_path = os.path.join(TMP_DIR, f"{filename_base}.jpg")
    output_dir = os.path.join(TMP_DIR, filename_base)
    os.makedirs(output_dir, exist_ok=True)

    # 保存上传文件
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 调用 RealESRGAN
    run_realesrgan_image(input_path, output_dir, suffix=unique_id)

    # 输出文件名加 _completed
    original_output_file = os.path.join(output_dir, f"{filename_base}_{unique_id}.png")
    completed_file = os.path.join(output_dir, f"{filename_base}_completed.png")

    if os.path.exists(original_output_file):
        os.rename(original_output_file, completed_file)
    else:
        raise RuntimeError(f"输出文件不存在: {original_output_file}")

    response = FileResponse(completed_file, filename=f"{filename_base}_completed.png")

    # 清理临时文件
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(input_path):
        os.remove(input_path)

    return response


@app.post("/superres-video")
async def superres_video(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    filename_base = f"{os.path.splitext(file.filename)[0]}_{unique_id}"
    input_path = os.path.join(TMP_DIR, f"{filename_base}.mp4")
    output_dir = os.path.join(TMP_DIR, filename_base)
    os.makedirs(output_dir, exist_ok=True)

    # 保存上传文件
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 调用 RealESRGAN 视频处理
    run_realesrgan_video(input_path, output_dir, suffix=unique_id)

    # 输出文件名加 _completed
    original_output_file = os.path.join(output_dir, f"{filename_base}_{unique_id}.mp4")
    completed_file = os.path.join(output_dir, f"{filename_base}_completed.mp4")

    if os.path.exists(original_output_file):
        os.rename(original_output_file, completed_file)
    else:
        raise RuntimeError(f"输出文件不存在: {original_output_file}")

    response = FileResponse(completed_file, filename=f"{filename_base}_completed.mp4")

    # 清理临时文件
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(input_path):
        os.remove(input_path)

    return response
