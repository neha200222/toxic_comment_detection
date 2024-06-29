import gradio as gr
import tensorflow as tf
from preprocess import load_and_preprocess_data
from inference.py import score_comment

model = tf.keras.models.load_model('models/toxicity.h5')
vectorizer = load_and_preprocess_data('data/train.csv')[2]

def score_comment_interface(comment):
    return score_comment(model, vectorizer, comment)

interface = gr.Interface(fn=score_comment_interface, inputs=gr.inputs.Textbox(lines=2, placeholder='Enter a comment'), outputs='text')
interface.launch(share=True)
