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

# Step 2: Summarize the paper abstract with adjustable length
def summarize_text(text, model_name="facebook/bart-large-cnn", max_len=300, min_len=100):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

# Step 3: Perform sentiment analysis on the summary
def analyze_sentiment(summary_text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(summary_text)
    return sentiment[0]

# Main function
def summarize_paper(paper_title, max_len=300, min_len=100):
    paper_info = search_paper(paper_title)
    summary = summarize_text(paper_info['abstract'], max_len=max_len, min_len=min_len)
    
    sentiment_result = analyze_sentiment(summary)  # Get the sentiment of the summary
    
    return {
        'title': paper_info['title'],
        'summary': summary,
        'url': paper_info['url'],
        'sentiment': sentiment_result  # Include sentiment in the result
    }

# Prompt user for the paper title
paper_title = input("Which paper do you want to summarize? ")

# Generate summary with user-provided paper title
result = summarize_paper(paper_title, max_len=300, min_len=100)

# Green color codes for summary
green_start = "\033[92m"
reset_color = "\033[0m"
magenta_color = "\033[35m"

# Print the result with the summary in green and sentiment analysis
print(f"Title: {result['title']}")
print(f"{green_start}Summary: {result['summary']}{reset_color}")
print(f"URL: {magenta_color}{result['url']}")

# Print the sentiment analysis result
print(f"Sentiment: {result['sentiment']['label']} (Confidence: {result['sentiment']['score']:.2f})")
