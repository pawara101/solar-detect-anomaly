import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
from PIL import Image
from pathlib import Path

from gradcam.eigencam import EigenCAM
from gradcam.utils import show_cam_on_image

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "yolo11bb_12_neck_v2best.pt"
def get_model():
    try:
        model = YOLO(MODEL_PATH)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


model = get_model()

# --------------------------------------------------
# Target layers for EigenCAM
# --------------------------------------------------
target_layers = [model.model.model[-3]]

# --------------------------------------------------
# Explainability function (EigenCAM)
# --------------------------------------------------
def explainModel(image_pil):
    # Convert PIL → OpenCV (RGB)
    img = np.array(image_pil)
    img = cv2.resize(img, (640, 640))

    rgb_img = img.copy()
    img_norm = np.float32(img) / 255.0

    cam = EigenCAM(model, target_layers, task="od")
    grayscale_cam = cam(rgb_img)[0]

    cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_image)

# --------------------------------------------------
# Detection + Explainability
# --------------------------------------------------
def detectAnomaly(image):
    try:
        start_time = time.perf_counter()

        results = model.predict(source=image, imgsz=640)

        end_time = time.perf_counter()
        logging.info(f"Inference time: {end_time - start_time:.3f}s")

        result = results[0]
        annotated_image_array = result.plot()

        # BGR → RGB
        annotated_image_array = annotated_image_array[:, :, ::-1]
        detected_image = Image.fromarray(annotated_image_array)

        # EigenCAM output
        cam_image = explainModel(image)

        return detected_image, cam_image

    except Exception as e:
        logging.error(f"Error during detection: {str(e)}")
        return image, None

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
with gr.Blocks(title="YOLO Object Detection System") as solarApp:

    gr.Markdown(
        """
        # 🔍 YOLO Object Detection & Explainability System
        Detect solar cell defects and visualize **why** the model made its decision.
        """
    )

    with gr.Tabs():

        # --------------------------------------------------
        # Home Tab
        # --------------------------------------------------
        with gr.Tab("🏠 Home"):
            gr.Markdown(
                """
                ## What does this system do?
                This application detects **defects in Solar Cell EL images**
                using a YOLO object detection model and explains predictions
                using **EigenCAM**.

                ## Why Explainability?
                - Builds trust in model decisions
                - Highlights important regions influencing detection
                - Useful for quality inspection and audits

                👉 Go to **Detection** to try it.
                """
            )

        # --------------------------------------------------
        # Detection Tab
        # --------------------------------------------------
        with gr.Tab("📷 Detection"):
            gr.Markdown("### Upload an image to run detection and explanation")

            with gr.Row():
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=480,
                    width=480
                )

            with gr.Row():
                output_image = gr.Image(
                    label="YOLO Detection Output",
                    height=480,
                    width=480
                )

                cam_output = gr.Image(
                    label="EigenCAM Explanation",
                    height=480,
                    width=480
                )

            detect_button = gr.Button("Run Detection & Explain")

            detect_button.click(
                fn=detectAnomaly,
                inputs=input_image,
                outputs=[output_image, cam_output]
            )

        # --------------------------------------------------
        # User Guide Tab
        # --------------------------------------------------
        with gr.Tab("📘 User Guide"):
            gr.Markdown(
                """
                ## How to Use

                1. Upload a Solar Cell EL image
                2. Click **Run Detection & Explain**
                3. View:
                   - **YOLO Detection** → detected defects
                   - **EigenCAM** → regions influencing predictions

                ## Interpretation Tips
                - Brighter CAM regions = higher model attention
                - CAM does not represent probability, only importance

                ## Limitations
                - CAM resolution is approximate
                - Depends on trained defect classes
                """
            )

# --------------------------------------------------
# Launch
# --------------------------------------------------
solarApp.launch()
