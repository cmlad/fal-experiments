import fal
from fal.toolkit import Image
from src.fal_handlers.sdxl_t2i.models import Input, Output
from src.pipelines.sdxl import get_pipeline

class SDXLT2I(fal.App):
    machine_type = "GPU"
    requirements = ["torch==2.7.0", "diffusers", "transformers"]

    def setup(self):

        self.pipeline = get_pipeline()

    @fal.endpoint("/text-to-image")
    def predict(self, input: Input) -> Output:
        images_pil = self.pipeline(
            prompt=input.prompt,
            num_inference_steps=30,
            width=1024,
            height=1024,
        ).images

        images_fal = [Image.from_pil(image) for image in images_pil]

        return Output(images=images_fal)
