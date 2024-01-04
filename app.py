from os.path import basename, splitext

import gradio as gr
from huggingface_hub import hf_hub_download

from onnx_inference import vectorize_image


MODEL_PATH = hf_hub_download("nopperl/marked-lineart-vectorizer", "model.onnx")


def predict(input_image_path, threshold, stroke_width):
    output_filepath = splitext(basename(input_image_path))[0] + ".svg"
    for recons_img in vectorize_image(input_image_path, model=MODEL_PATH, output=output_filepath, threshold_ratio=threshold, stroke_width=stroke_width):
        yield recons_img
    yield output_filepath


interface = gr.Interface(
        predict,
        inputs=[gr.Image(sources="upload", type="filepath"), gr.Slider(minimum=0.1, maximum=0.9, value=0.1, label="threshold"), gr.Slider(minimum=0.1, maximum=4.0, value=0.512, label="stroke_width")],
        outputs=gr.Image(),
        description="Demo for a model that converts raster line-art images into vector images iteratively. The model is trained on black-and-white line-art images, hence it won't work with other images. Inference time will be quite slow due to a lack of GPU resources. More information at https://github.com/nopperl/marked-lineart-vectorization.",
        examples = [
            ["examples/01.png", 0.1, 0.512],
            ["examples/02.png", 0.1, 0.512]
        ],
        analytics_enabled=False
    )
interface.launch()
