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
    summary = summarizer(text, max_length=300, min_length=150, do_sample=False)
    return summary[0]['summary_text']

# Main function
def summarize_paper(paper_title):
    paper_info = search_paper(paper_title)
    summary = summarize_text(paper_info['abstract'])
    
    return {
        'title': paper_info['title'],
        'summary': summary,
        'url': paper_info['url']
    }

# Example usage
paper_title = "Neural networks for machine learning"
result = summarize_paper(paper_title)

# Green color codes
green_start = "\033[92m"
reset_color = "\033[0m"

# Print the result with the summary in green
print(f"Title: {result['title']}")
print(f"{green_start}Summary: {result['summary']}{reset_color}")
print(f"URL: {result['url']}")
