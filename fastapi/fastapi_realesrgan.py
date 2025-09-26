import os
import uuid
import shutil
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
INPUT_DIR = "datasets/simple_test"
OUTPUT_DIR = "results/api_test"
MODEL_PATH = (
    "experiments/pretrained_models/Real-ESRGAN/RealESRGAN_x4plus.pth"
)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


def run_esrgan(input_path, output_dir):
    cmd = [
        "python",
        "inference/inference_realesrgan.py",
        "--input",
        input_path,
        "--output",
        output_dir,
        "--model_path",
        MODEL_PATH,
    ]
    subprocess.run(cmd, check=True)


@app.post("/esrgan")
async def esrgan_infer(
    image: UploadFile = File(...), background_tasks: BackgroundTasks = None # type: ignore
):
    filename = f"{uuid.uuid4().hex}.png"
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # 保存上传图片
    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, run_esrgan, input_path, OUTPUT_DIR)
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "ESRGAN inference failed", "detail": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": "Unknown error", "detail": str(e)}
        )

    if not os.path.exists(output_path):
        return JSONResponse(status_code=500, content={"error": "Output not found"})

    # 响应返回后，后台清理文件
    background_tasks.add_task(os.remove, input_path)
    background_tasks.add_task(os.remove, output_path)

    return FileResponse(output_path, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok"}
