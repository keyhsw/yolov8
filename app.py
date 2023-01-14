import gradio as gr
import torch
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import visualize_object_predictions, read_image
from ultralyticsplus import YOLO

# Images
torch.hub.download_url_to_file('https://raw.githubusercontent.com/kadirnar/dethub/main/data/images/highway.jpg', 'highway.jpg')
torch.hub.download_url_to_file('https://user-images.githubusercontent.com/34196005/142742872-1fefcc4d-d7e6-4c43-bbb7-6b5982f7e4ba.jpg', 'highway1.jpg')
torch.hub.download_url_to_file('https://raw.githubusercontent.com/obss/sahi/main/tests/data/small-vehicles1.jpeg', 'small-vehicles1.jpeg')

def yolov8_inference(
    image: gr.inputs.Image = None,
    model_path: gr.inputs.Dropdown = None,
    image_size: gr.inputs.Slider = 640,
    conf_threshold: gr.inputs.Slider = 0.25,
    iou_threshold: gr.inputs.Slider = 0.45,
):
    """
    YOLOv8 inference function
    Args:
        image: Input image
        model_path: Path to the model
        image_size: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
    Returns:
        Rendered image
    """
    model = YOLO(model_path)
    model.conf = conf_threshold
    model.iou = iou_threshold
    results = model.predict(image, imgsz=image_size, return_outputs=True)
    object_prediction_list = []
    for _, image_results in enumerate(results):
        image_predictions_in_xyxy_format = image_results['det']
        for pred in image_predictions_in_xyxy_format:
            x1, y1, x2, y2 = (
                int(pred[0]),
                int(pred[1]),
                int(pred[2]),
                int(pred[3]),
            )
            bbox = [x1, y1, x2, y2]
            score = pred[4]
            category_name = model.model.names[int(pred[5])]
            category_id = pred[5]
            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=int(category_id),
                score=score,
                category_name=category_name,
            )
            object_prediction_list.append(object_prediction)

    image = read_image(image)
    output_image = visualize_object_predictions(image=image, object_prediction_list=object_prediction_list)
    return output_image['image']
        

inputs = [
    gr.inputs.Image(type="filepath", label="Input Image"),
    gr.inputs.Dropdown(["kadirnar/yolov8n-v8.0", "kadirnar/yolov8m-v8.0", "kadirnar/yolov8l-v8.0", "kadirnar/yolov8x-v8.0", "kadirnar/yolov8x6-v8.0"], 
                       default="kadirnar/yolov8m-v8.0", label="Model"),
    gr.inputs.Slider(minimum=320, maximum=1280, default=640, step=32, label="Image Size"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.25, step=0.05, label="Confidence Threshold"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.outputs.Image(type="filepath", label="Output Image")
title = "Ultralytics YOLOv8: State-of-the-Art YOLO Models"

examples = [['highway.jpg', 'kadirnar/yolov8m-v8.0', 640, 0.25, 0.45], ['highway1.jpg', 'kadirnar/yolov8l-v8.0', 640, 0.25, 0.45], ['small-vehicles1.jpeg', 'kadirnar/yolov8x-v8.0', 1280, 0.25, 0.45]]
demo_app = gr.Interface(
    fn=yolov8_inference,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
    theme='huggingface',
)
demo_app.launch(debug=True, enable_queue=True)