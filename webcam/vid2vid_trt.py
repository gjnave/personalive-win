import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

from multiprocessing import Queue, Manager, Event, Process
from .util import read_images_from_queue, image_to_array, array_to_image, clear_queue

import time
from typing import List
import torch
from .config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math
from src.wrapper_trt import PersonaLive


page_content = """<h1 class="text-3xl font-bold">🎭 PersonaLive!</h1>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/GVCLab/PersonaLive"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">PersonaLive
</a>
video-to-video pipeline with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "PersonaLive"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )

    def __init__(self, args: Args, device: torch.device):
        self.args = args
        self.device = device
        self.prepare()

    def prepare(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.reference_queue = Queue()

        self.prepare_event = Event()
        self.stop_event = Event()
        self.restart_event = Event()

        self.process = Process(
            target=generate_process,
            args=(self.args.config_path, self.prepare_event, self.restart_event, self.stop_event, self.input_queue, self.output_queue, self.reference_queue, self.device),
            daemon=True
        )
        self.process.start()
        self.processes = [self.process]
        self.prepare_event.wait()

    def accept_new_params(self, params: "Pipeline.InputParams"):
        if hasattr(params, "image"):
            # Keep on CPU for multiprocessing queue transfer
            image_tensor = params.image.float() / 255.0
            image_tensor = image_tensor * 2. - 1.
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            self.input_queue.put(image_tensor)

        if hasattr(params, "restart") and params.restart:
            self.restart_event.set()
            clear_queue(self.output_queue)

    def fuse_reference(self, ref_image):
        self.reference_queue.put(ref_image)

    def produce_outputs(self) -> List[Image.Image]:
        qsize = self.output_queue.qsize()
        results = []
        for _ in range(qsize):
            results.append(array_to_image(self.output_queue.get()))
            # results.append(self.output_queue.get())
        return results

    def close(self):
        print("Setting stop event...")
        self.stop_event.set()

        print("Waiting for processes to terminate...")
        for i, process in enumerate(self.processes):
            process.join(timeout=1.0)
            if process.is_alive():
                print(f"Process {i} didn't terminate gracefully, forcing termination")
                process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    print(f"Force killing process {i}")
                    process.kill()
        print("Pipeline closed successfully")

def generate_process(
        config_path,
        prepare_event,
        restart_event,
        stop_event,
        input_queue,
        output_queue,
        reference_queue,
        device):
    import os
    # Enable synchronous CUDA errors for better debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.set_grad_enabled(False)

    # Initialize CUDA context in this process
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        # Force CUDA initialization by allocating a small tensor
        _ = torch.zeros(1, device=device)
        print(f"CUDA initialized in worker process on device: {device}")

    pipeline = PersonaLive(config_path, device)

    # After TensorRT initialization, ensure PyTorch CUDA context is active
    if device.type == 'cuda':
        print("Testing PyTorch CUDA context after TensorRT initialization...")
        try:
            torch.cuda.set_device(device)
            torch.cuda.synchronize()
            # Test that PyTorch CUDA works
            test_tensor = torch.randn(1, 3, 64, 64, device=device)
            print(f"  Created test tensor on {test_tensor.device}")
            test_resized = torch.nn.functional.interpolate(test_tensor, size=(32, 32), mode='bilinear')
            print(f"  Interpolation successful: {test_resized.shape}")
            print(f"✓ PyTorch CUDA context validated successfully")
            del test_tensor, test_resized
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ PyTorch CUDA validation FAILED: {e}")
            print("  This indicates TensorRT initialization corrupted PyTorch's CUDA context")
            import traceback
            traceback.print_exc()
            raise
    chunk_size = 4

    prepare_event.set()

    # Wait for initial reference image
    reference_img = reference_queue.get()
    pipeline.fuse_reference(reference_img)
    print('Initial reference fused')

    while not stop_event.is_set():
        # Check if there's a new reference image to fuse
        if not reference_queue.empty():
            try:
                new_reference = reference_queue.get_nowait()
                print('New reference image detected, fusing...')
                pipeline.fuse_reference(new_reference)
                print('New reference fused successfully')
            except Exception as e:
                print(f'Failed to get new reference: {e}')

        if restart_event.is_set():
            clear_queue(input_queue)
            restart_event.clear()
        images = read_images_from_queue(input_queue, chunk_size, device, stop_event)
        if images is None:
            continue

        try:
            # Concatenate on CPU first
            images = torch.cat(images, dim=0)
            # Ensure contiguous memory layout
            images = images.contiguous()

            # Debug before GPU transfer
            print(f"Before GPU transfer - Shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
            print(f"  Data range check - min: {images.min().item():.4f}, max: {images.max().item():.4f}")
            print(f"  Has NaN: {torch.isnan(images).any().item()}, Has Inf: {torch.isinf(images).any().item()}")

            # Move to GPU with explicit synchronization
            images = images.to(device, non_blocking=False)
            torch.cuda.synchronize()

            print(f"After GPU transfer - Shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")

            video = pipeline.process_input(images)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error encountered: {e}")
                print("CUDA context may be corrupted. Attempting recovery...")
                # Try to reset CUDA context
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # If still failing, break the loop to prevent infinite errors
                print("Stopping processing loop due to persistent CUDA errors")
                break
            else:
                print(f"Error processing images: {e}")
                import traceback
                traceback.print_exc()
                continue
        except Exception as e:
            print(f"Unexpected error processing images: {e}")
            import traceback
            traceback.print_exc()
            continue
        for image in video:
            output_queue.put(image)