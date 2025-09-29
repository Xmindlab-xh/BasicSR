import os
import shutil
import uuid
import subprocess
import logging
import asyncio
import threading
import time
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    ]
    if suffix:
        cmd.extend(["--suffix", suffix])
    logger.info(f"执行命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 记录详细的输出信息
    logger.info(f"命令返回码: {result.returncode}")
    if result.stdout:
        logger.info(f"标准输出: {result.stdout}")
    if result.stderr:
        logger.info(f"标准错误: {result.stderr}")

    if result.returncode != 0:
        error_msg = f"RealESRGAN 执行失败 (返回码: {result.returncode})\n标准错误: {result.stderr}\n标准输出: {result.stdout}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return result

def run_realesrgan_video(input_path: str, output_dir: str, model_name="realesr-animevideov3", suffix: str | None = None):
    # 检查输入文件
    if not os.path.exists(input_path):
        raise RuntimeError(f"输入文件不存在: {input_path}")

    file_size = os.path.getsize(input_path)
    logger.info(f"输入文件大小: {file_size / 1024 / 1024:.2f} MB")

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
    logger.info(f"执行命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 记录详细的输出信息
    logger.info(f"命令返回码: {result.returncode}")
    if result.stdout:
        logger.info(f"标准输出: {result.stdout}")
    if result.stderr:
        logger.info(f"标准错误: {result.stderr}")

    # 检查输出目录中的文件（无论返回码如何）
    if os.path.exists(output_dir):
        files_after = os.listdir(output_dir)
        logger.info(f"处理后输出目录中的文件: {files_after}")

        # 检查是否有临时目录
        temp_video_dir = None
        for f in files_after:
            if f.endswith('_out_tmp_videos') and os.path.isdir(os.path.join(output_dir, f)):
                temp_video_dir = os.path.join(output_dir, f)
                break

        if temp_video_dir:
            temp_files = os.listdir(temp_video_dir)
            logger.info(f"临时视频目录 {temp_video_dir} 中的文件: {temp_files}")

    # 即使返回码为0，如果stderr中包含Error，也认为是失败
    if result.returncode != 0 or (result.stderr and "Error" in result.stderr):
        error_msg = f"RealESRGAN 视频处理失败 (返回码: {result.returncode})\n标准错误: {result.stderr}\n标准输出: {result.stdout}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return result

def cleanup_files_delayed(input_path: str, output_dir: str, delay: int = 10):
    """延迟清理文件，给文件传输留出时间"""
    def cleanup():
        time.sleep(delay)  # 延迟指定秒数
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"已删除输入文件: {input_path}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logger.info(f"已删除输出目录: {output_dir}")
            logger.info("延迟清理完成")
        except Exception as e:
            logger.error(f"延迟清理文件时出错: {e}")

    # 在后台线程中执行清理
    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.daemon = True  # 设为守护线程
    cleanup_thread.start()

def find_output_file(output_dir: str, original_filename: str, suffix: str):
    """查找实际生成的输出文件"""
    # 可能的输出文件名格式
    base_name = os.path.splitext(original_filename)[0]
    ext = os.path.splitext(original_filename)[1]

    possible_names = [
        f"{base_name}_{suffix}_out{ext}",  # 常见格式: filename_suffix_out.ext
        f"{base_name}_out{ext}",          # 格式: filename_out.ext
        f"{base_name}_{suffix}{ext}",     # 格式: filename_suffix.ext
        f"{base_name}{ext}",              # 原文件名
    ]

    logger.info(f"查找输出文件，目录: {output_dir}")
    logger.info(f"可能的文件名: {possible_names}")

    # 列出输出目录中的所有文件进行调试
    if os.path.exists(output_dir):
        files_in_dir = os.listdir(output_dir)
        logger.info(f"输出目录中的文件: {files_in_dir}")

        # 先检查预期的文件名
        for name in possible_names:
            full_path = os.path.join(output_dir, name)
            if os.path.exists(full_path):
                logger.info(f"找到输出文件: {full_path}")
                return full_path

        # 如果找不到预期文件名，返回目录中第一个文件（如果有的话）
        if files_in_dir:
            # 过滤掉可能的输入文件，只要处理后的文件
            processed_files = [f for f in files_in_dir if not f.endswith('_input' + ext)]
            if processed_files:
                found_file = os.path.join(output_dir, processed_files[0])
                logger.info(f"使用找到的文件: {found_file}")
                return found_file

    return None

@app.post("/superres-image")
async def superres_image(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    ext = os.path.splitext(file.filename)[1].lower()
    original_name = os.path.splitext(file.filename)[0]

    # 输入文件路径
    input_filename = f"{original_name}_{unique_id}{ext}"
    input_path = os.path.join(TMP_DIR, input_filename)

    # 输出目录
    output_dir = os.path.join(TMP_DIR, f"output_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"处理图片: {file.filename}")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {output_dir}")

    try:
        # 保存上传的文件
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 执行 RealESRGAN
        run_realesrgan_image(input_path, output_dir, suffix=unique_id)

        # 查找实际生成的输出文件
        output_file = find_output_file(output_dir, file.filename, unique_id)

        if not output_file or not os.path.exists(output_file):
            raise HTTPException(status_code=500, detail=f"找不到输出文件，输出目录: {output_dir}")

        # 重命名为最终文件
        final_filename = f"{original_name}_enhanced{ext}"
        final_path = os.path.join(output_dir, final_filename)

        if output_file != final_path:
            os.rename(output_file, final_path)

        logger.info(f"处理完成，返回文件: {final_path}")

        # 启动延迟清理任务
        cleanup_files_delayed(input_path, output_dir, delay=30)  # 30秒后清理

        # 返回文件
        response = FileResponse(
            final_path,
            filename=final_filename
        )

        return response

    except Exception as e:
        logger.error(f"处理图片时出错: {str(e)}")
        # 清理文件
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"图片处理失败: {str(e)}")

@app.post("/superres-video")
async def superres_video(file: UploadFile = File(...)):
    unique_id = get_unique_name()
    ext = os.path.splitext(file.filename)[1].lower()
    original_name = os.path.splitext(file.filename)[0]

    # 输入文件路径
    input_filename = f"{original_name}_{unique_id}{ext}"
    input_path = os.path.join(TMP_DIR, input_filename)

    # 输出目录
    output_dir = os.path.join(TMP_DIR, f"output_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"处理视频: {file.filename}")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {output_dir}")

    try:
        # 保存上传的文件
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 执行 RealESRGAN
        run_realesrgan_video(input_path, output_dir, suffix=unique_id)

        # 查找实际生成的输出文件
        output_file = find_output_file(output_dir, file.filename, unique_id)

        if not output_file or not os.path.exists(output_file):
            raise HTTPException(status_code=500, detail=f"找不到输出文件，输出目录: {output_dir}")

        # 重命名为最终文件
        final_filename = f"{original_name}_enhanced{ext}"
        final_path = os.path.join(output_dir, final_filename)

        if output_file != final_path:
            os.rename(output_file, final_path)

        logger.info(f"处理完成，返回文件: {final_path}")

        # 启动延迟清理任务
        cleanup_files_delayed(input_path, output_dir, delay=30)  # 30秒后清理

        # 返回文件
        response = FileResponse(
            final_path,
            filename=final_filename
        )

        return response

    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        # 清理文件
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}