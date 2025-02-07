# # Libraries

# from PIL import Image
# from io import BytesIO
# import pytesseract
# # Specify the path where Tesseract-OCR was installed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# import pandas as pd
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from pytesseract import Output
# import re
# import glob
# import os
# import PIL.Image
# from PIL import Image
# from pdf2image import convert_from_path
# from PyPDF2 import PdfReader
# from PyPDF2.errors import PdfReadError
import os
import json
import getpass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.tools import tool
from langchain_openai import ChatOpenAI
# from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

import openai
from openai import OpenAI
import re

def find_relevant_chunks_tfidf_2step(question, directory_path, max_files=3, max_chunks=5):
    """
    A 2-step TF-IDF approach:
      1) Compare the question with the "summary" of each JSON file,
         select the top 'max_files' relevant files.
      2) For each selected file, compare the question with the text of each chunk
         in 'contentList', and pick the top 'max_chunks'.

    Returns:
        A dict with:
          - "relevant_files": list of file-level matches (sorted by similarity desc)
          - "relevant_chunks": list of chunk-level matches across those files
    """

    # --------------------------------------------------------------------
    # STEP 1: File-level search by comparing question vs. "summary"
    # --------------------------------------------------------------------
    files_data = []  # Will store (filename, summary_text, content_list)
    for file_name in os.listdir(directory_path):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(directory_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Each file has:
                # {
                #   "summary": "...",
                #   "contentList": [{ "content": ... }, ...]
                # }
                summary_text = data.get("summary", "").strip()
                content_list = data.get("contentList", [])
                files_data.append((file_name, summary_text, content_list))
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # If no files or all empty, return empty results
    if not files_data:
        return {
            "relevant_files": [],
            "relevant_chunks": []
        }

    # Build TF-IDF for the question + all summaries
    summaries = [fd[1] for fd in files_data]  # all summary texts
    documents = [question] + summaries        # index 0 is the question
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    question_vector = tfidf_matrix[0]    # the question
    summary_vectors = tfidf_matrix[1:]   # the file summaries
    similarities = cosine_similarity(question_vector, summary_vectors).flatten()

    # Pair each file with its similarity
    file_scores = list(zip(similarities, files_data))
    # Sort by similarity descending
    file_scores.sort(key=lambda x: x[0], reverse=True)

    # Pick top N files
    top_files = file_scores[:max_files]

    # Format them for returning
    relevant_files = [
        {
            "file_name": file_info[0],
            "summary": file_info[1],
            "similarity": sim
        }
        for (sim, file_info) in top_files
    ]

    # --------------------------------------------------------------------
    # STEP 2: Within each top file, compare question vs. each chunk
    # --------------------------------------------------------------------
    all_relevant_chunks = []
    for sim_file, (file_name, summary_text, content_list) in top_files:
        # Build a new doc set: [question] + all chunk texts
        chunk_texts = [chunk.get("content", "") for chunk in content_list if chunk.get("content")]
        if not chunk_texts:
            continue  # no chunks in this file

        documents = [question] + chunk_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        question_vector = tfidf_matrix[0]
        chunk_vectors = tfidf_matrix[1:]
        chunk_sims = cosine_similarity(question_vector, chunk_vectors).flatten()

        # Pair each chunk with its similarity
        chunk_scores = list(zip(chunk_sims, content_list))
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        top_chunks = chunk_scores[:max_chunks]

        # Format chunk-level results
        for chunk_sim, chunk_data in top_chunks:
            all_relevant_chunks.append({
                "file_name": file_name,
                "similarity": chunk_sim,
                "chunk": chunk_data
            })

    # Return both file-level results and chunk-level results
    return {
        "relevant_files": relevant_files,
        "relevant_chunks": all_relevant_chunks
    }

@tool
def search_relevant_chunks_tool_new(question: str, directory_path: str, max_files: int = 3, max_chunks: int = 5) -> dict:

    """
    Searches for the most relevant JSON files in a directory based on their 'summary',
    then for each chosen file, selects the most relevant chunks from 'contentList'.

    Args:
    - question (str): The question or query to compare against each JSON file's summary and chunks.
    - directory_path (str): Path to the directory containing JSON files with structure:
            {
                "summary": "...",
                "contentList": [
                    { "content": "..." },
                    ...
                ]
            }
    - max_files (int): How many of the top relevant files to retrieve by summary-level matching.
    - max_chunks (int): How many top relevant chunks to retrieve from each file.

    Returns:
    - dict: A dictionary with two keys. For "relevant_files", the values are lists of dictionaries that contain the file name, summary, and similarity. For "relevant_chunks", the values are lists of dictionaries that contain the file name, similarity, and chunk content.
    """
    return find_relevant_chunks_tfidf_2step(question, directory_path, max_files, max_chunks)
