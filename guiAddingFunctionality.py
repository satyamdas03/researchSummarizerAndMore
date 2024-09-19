import customtkinter as ctk
import requests
from scholarly import scholarly
from transformers import pipeline

# Step 1: Fetch paper details from Google Scholar
def search_paper(paper_title):
    search_query = scholarly.search_pubs(paper_title)
    paper = next(search_query)
    return {
        'title': paper['bib']['title'],
        'abstract': paper['bib']['abstract'],
        'url': paper['pub_url'] if 'pub_url' in paper else "No URL found"
    }

# Step 2: Summarize the paper abstract
def summarize_text(text, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Function to handle the summary generation
def summarize_paper():
    paper_title = title_entry.get()
    paper_info = search_paper(paper_title)
    summary = summarize_text(paper_info['abstract'])
    
    result_title.config(text=f"Title: {paper_info['title']}")
    result_summary.config(text=f"Summary: {summary}")
    result_url.config(text=f"URL: {paper_info['url']}")

# Set up the CustomTkinter GUI
app = ctk.CTk()
app.title("Research Paper Summarizer")

# Create and place widgets
title_entry = ctk.CTkEntry(app, placeholder_text="Enter paper title")
title_entry.pack(pady=10)

summarize_button = ctk.CTkButton(app, text="Summarize", command=summarize_paper)
summarize_button.pack(pady=10)

result_title = ctk.CTkLabel(app, text="Title: ")
result_title.pack(pady=5)

result_summary = ctk.CTkLabel(app, text="Summary: ")
result_summary.pack(pady=5)

result_url = ctk.CTkLabel(app, text="URL: ")
result_url.pack(pady=5)

app.mainloop()
