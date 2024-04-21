import gradio as gr
from transformers import pipeline

pipe = pipeline("text-classification", "ecabott/nepali-sentiment-analysis")

def text_classifier(text):
    outputs = pipe(text)
    for result in outputs:
        if result['label'] == 'LABEL_1':
            return f"Positive sentiment Score:{result['score']:.3f}"
        if result['label'] == 'LABEL_2':
            return f"Neutral Sentiment Score:{result['score']:.3f}"
        else:
            return f"Negative Sentiment Score: {result['score']:.3f}"

title = "Nepali Sentiment Analysis"
description = """
This app is a demonstration of using a BERT model(NepBERTa) to classify sentiment of a Nepali Sentence.\n
More information:\n
https://github.com/ecabott/nepali-sentiment-analysis\n
https://huggingface.co/ecabott/nepali-sentiment-analysis\n
"""

demo = gr.Interface(fn=text_classifier, inputs=gr.Text(type="text"), outputs="label", title=title, description=description)
demo.launch(show_api=False)