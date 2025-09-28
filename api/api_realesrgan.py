import os
import shutil
import uuid
import subprocess
import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# ----------------- 日志配置 -----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("superres")
logger.setLevel(logging.INFO)

# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 文件 handler，按天生成日志
file_handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_DIR, "superres.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
    utc=False
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
file_handler.suffix = "%Y-%m-%d.log"
logger.addHandler(file_handler)
# -------------------------------------------

app = FastAPI()

TMP_DIR = "temp"
os.makedirs(TMP_DIR, exist_ok=True)

def get_unique_name():
    return str(uuid.uuid4())

def run_realesrgan_image(input_path: str, output_dir: str, model_name="RealESRGAN_x4plus", suffix=""):
    cmd = [
        "python",
        "inference/inference_realesrgan.py",
        "-i", input_path,
        "-o", output_dir,
        "-n", model_name,
        "--suffix", suffix
    ]
    subprocess.run(cmd, check=True)

def run_realesrgan_video(input_path: str, output_dir: str, model_name="realesr-animevideov3", suffix=""):
    num_process = 1
    cmd = [
        "python",
        "inference/inference_realesrgan_video.py",
        "-i", input_path,
        "-o", output_dir,
        "-n", model_name,
        "--suffix", suffix,
        "--num_process", str(num_process)
    ]
    subprocess.run(cmd, check=True)

@app.post("/superres-image")
async def superres_image(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    ext = os.path.splitext(file.filename)[1]  # 保留原始后缀
    filename_base = f"{os.path.splitext(file.filename)[0]}_{unique_id}"
    input_path = os.path.join(TMP_DIR, f"{filename_base}{ext}")
    output_dir = os.path.join(TMP_DIR, filename_base)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    run_realesrgan_image(input_path, output_dir, suffix=unique_id)
    logger.info(f"input_path: {input_path}\noutput_dir：{output_dir}")

    # 输出文件名加 _completed
    original_output_file = os.path.join(output_dir, f"{filename_base}{ext}")  # RealESRGAN 输出默认带 suffix
    completed_file = os.path.join(output_dir, f"{filename_base}_completed{ext}")
    logger.info(f"original_output_file: {original_output_file}\completed_file{completed_file}")

    if not os.path.exists(original_output_file):
        raise RuntimeError(f"输出文件不存在: {original_output_file}")

    os.rename(original_output_file, completed_file)

    response = FileResponse(completed_file, filename=os.path.basename(completed_file))

    # 清理临时文件
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(input_path):
        os.remove(input_path)

    return response

@app.post("/superres-video")
async def superres_video(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    ext = os.path.splitext(file.filename)[1]  # 保留原始后缀
    filename_base = f"{os.path.splitext(file.filename)[0]}_{unique_id}"
    input_path = os.path.join(TMP_DIR, f"{filename_base}{ext}")
    output_dir = os.path.join(TMP_DIR, filename_base)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    run_realesrgan_video(input_path, output_dir, suffix=unique_id)

    # 输出文件名加 _completed
    original_output_file = os.path.join(output_dir, f"{filename_base}{ext}")  # RealESRGAN 输出默认带 suffix
    completed_file = os.path.join(output_dir, f"{filename_base}_completed{ext}")

    if not os.path.exists(original_output_file):
        raise RuntimeError(f"输出文件不存在: {original_output_file}")

    os.rename(original_output_file, completed_file)

    response = FileResponse(completed_file, filename=os.path.basename(completed_file))

    # 清理临时文件
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists(input_path):
        os.remove(input_path)

    return response
