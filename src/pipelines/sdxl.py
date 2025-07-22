def get_pipeline(device: str = "cuda"):
    from diffusers import DiffusionPipeline
    import torch

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    return pipe.to(device)

