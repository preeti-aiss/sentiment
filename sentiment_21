import streamlit as st
from transformers import pipeline

# Caching the model to prevent reloading on every interaction
@st.cache_resource()
def load_model():
    # Load a sentiment analysis model fine-tuned on the SST-2 dataset
    model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return model

# Load the sentiment analysis model
model = load_model()

# Set up the Streamlit interface
st.title('Sentiment Analyzer - Using DistilBERT')
st.header('Enter text to analyze:')

# Text input area
user_input = st.text_area('Write something to analyze:', height=200)

# Sentiment analysis when the button is clicked
if st.button('Analyze Sentiment'):
    if user_input.strip() != "":  # Check if input is not empty
        with st.spinner('Analyzing Sentiment...'):
            response = model(user_input)
            for i, result in enumerate(response):
                st.write(f'**Text {i+1}:**')
                st.write(f"Sentiment: {result['label']}")
                st.write(f"Confidence Score: {result['score']:.4f}")
                st.markdown("---")
    else:
        st.error("Please enter some text to analyze.")

# Sidebar with instructions
st.sidebar.markdown("## Guide")
st.sidebar.info(
    "This tool uses a transformer model (DistilBERT) fine-tuned for sentiment analysis. "
    "It will classify the input text as either positive or negative sentiment. "
    "Simply input your text and hit 'Analyze Sentiment' to see the result."
)
st.sidebar.markdown("### Examples")
st.sidebar.write("1. 'I love this product!' -> Positive")
st.sidebar.write("2. 'This is the worst experience ever.' -> Negative")
