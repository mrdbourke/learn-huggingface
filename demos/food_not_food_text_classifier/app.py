import torch
import gradio as gr

from transformers import pipeline

def food_not_food_classifier(text):
    # Set up text classification pipeline
    food_not_food_classifier = pipeline(task="text-classification", 
                                        model="mrdbourke/learn_hf_food_not_food_text_classifier-distilbert-base-uncased",
                                        device="cuda" if torch.cuda.is_available() else "cpu",
                                        top_k=None) # return all possible scores (not just top-1)
    
    # Get outputs from pipeline (as a list of dicts)
    outputs = food_not_food_classifier(text)[0]

    # Format output for Gradio (e.g. {"label_1": probability_1, "label_2": probability_2})
    output_dict = {}
    for item in outputs:
        output_dict[item["label"]] = item["score"]

    return output_dict

description = """
A text classifier to determine if a sentence is about food or not food.

TK - See source code:
"""

demo = gr.Interface(fn=food_not_food_classifier, 
             inputs="text", 
             outputs=gr.Label(num_top_classes=2), # show top 2 classes (that's all we have)
             title="🍗🚫🥑 Food or Not Food Text Classifier",
             description=description,
             examples=[["I whipped up a fresh batch of code, but it seems to have a syntax error."],
                       ["A delicious photo of a plate of scrambled eggs, bacon and toast."]])

if __name__ == "__main__":
    demo.launch()
