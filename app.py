import gradio as gr
from fastai.vision.all import *

# Re-define the GrayscaleTransform class (same as during training)
class GrayscaleTransform(Transform):
    def encodes(self, img: PILImage):
        return img.convert("L")  # Convert to grayscale

# Load the trained model
learn = load_learner('export.pkl')

# Define the prediction function
def classify_waterfowl(img):
    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Create a Gradio interface
interface = gr.Interface(fn=classify_waterfowl, inputs=gr.Image(type="pil"), outputs=gr.Label())

# Launch the Gradio app
interface.launch()
