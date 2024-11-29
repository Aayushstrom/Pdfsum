#!/usr/bin/env python
# coding: utf-8

# In[3]:


#get_ipython().run_line_magic('pip', 'install PyPDF2')


# In[1]:


import re
from PyPDF2 import PdfReader

def preprocess_text(text):
    """
    Clean and preprocess text by removing unwanted characters and extra spaces.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s.,;!?]', '', text)  # Remove non-alphanumeric characters
    return text.strip()


# In[2]:


def split_text(text, max_chunk_size=1000):
    """
    Split text into chunks within the token limit.
    """
    words = text.split()
    for i in range(0, len(words), max_chunk_size):
        yield " ".join(words[i:i + max_chunk_size])


# In[3]:


from transformers import pipeline

# Load the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_chunks(text_chunks, max_length=400, min_length=30):
    """
    Summarize each chunk and combine the summaries.
    """
    summaries = []
    for i, chunk in enumerate(text_chunks):
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
    return " ".join(summaries)


# In[ ]:


def extract_and_summarize(pdf_file):
    """
    Extract text from a PDF file, preprocess, and summarize.
    """
    # Step 1: Read the PDF
    reader = PdfReader(pdf_file)
    document_text = ""
    for page in reader.pages:
        document_text += page.extract_text()

    # Step 2: Preprocess the text
    cleaned_text = preprocess_text(document_text)

    # Step 3: Split the text into chunks
    text_chunks = list(split_text(cleaned_text, max_chunk_size=1000))

    # Step 4: Summarize each chunk
    summary = summarize_chunks(text_chunks)
    return summary


# In[5]:


import streamlit as st

# Streamlit interface
st.title("Intelligent PDF Summarizer")
st.write("Upload a large PDF document to summarize its content.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing..."):
        try:
            # Extract and summarize the PDF content
            summary = extract_and_summarize(uploaded_file)
            st.success("Summarization complete!")
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")


# In[ ]:
