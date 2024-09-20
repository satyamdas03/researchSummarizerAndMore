import requests
from scholarly import scholarly
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import defaultdict

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

# Step 4: Summarize and analyze a single paper
def summarize_paper(paper_title, max_len=300, min_len=100):
    paper_info = search_paper(paper_title)
    summary = summarize_text(paper_info['abstract'], max_len=max_len, min_len=min_len)
    sentiment_result = analyze_sentiment(summary)  # Get the sentiment of the summary
    return {
        'title': paper_info['title'],
        'summary': summary,
        'url': paper_info['url'],
        'sentiment': sentiment_result['label'],  # Include sentiment in the result
        'confidence': sentiment_result['score']
    }

# Step 5: Multiple paper comparison and visualization
def compare_papers(paper_titles):
    paper_results = []
    
    for title in paper_titles:
        result = summarize_paper(title)
        paper_results.append(result)

    # Print summaries and sentiment results
    for result in paper_results:
        print(f"Title: {result['title']}")
        print(f"Summary: {result['summary']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print(f"URL: {result['url']}\n")

    # Visualize sentiment comparison with a bar chart
    visualize_comparison(paper_results)

# Step 6: Visualization of key findings and sentiment comparison
def visualize_comparison(paper_results):
    titles = [result['title'] for result in paper_results]
    sentiments = [result['confidence'] for result in paper_results]

    plt.figure(figsize=(10, 6))
    plt.barh(titles, sentiments, color=['green' if result['sentiment'] == 'POSITIVE' else 'red' for result in paper_results])
    plt.xlabel('Sentiment Confidence')
    plt.ylabel('Research Papers')
    plt.title('Sentiment Comparison Between Papers')
    plt.tight_layout()
    plt.show()

# Main execution for paper comparison
if __name__ == "__main__":
    # Prompt user for multiple paper titles
    paper_titles = input("Enter paper titles (comma-separated) to summarize and compare: ").split(',')

    # Trim any leading or trailing whitespace from the paper titles
    paper_titles = [title.strip() for title in paper_titles]

    # Generate comparison and visualization for the given paper titles
    compare_papers(paper_titles)
