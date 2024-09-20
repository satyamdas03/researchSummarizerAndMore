import requests
from scholarly import scholarly
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from matplotlib_venn import venn2, venn3  # Import Venn diagram tools

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

# Step 4: Extract keywords from the paper abstract
def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    return keywords

# Step 5: Summarize and analyze a single paper
def summarize_paper(paper_title, max_len=300, min_len=100):
    paper_info = search_paper(paper_title)
    summary = summarize_text(paper_info['abstract'], max_len=max_len, min_len=min_len)
    sentiment_result = analyze_sentiment(summary)  # Get the sentiment of the summary
    keywords = extract_keywords(paper_info['abstract'])
    return {
        'title': paper_info['title'],
        'summary': summary,
        'url': paper_info['url'],
        'sentiment': sentiment_result['label'],  # Include sentiment in the result
        'confidence': sentiment_result['score'],
        'abstract_length': len(paper_info['abstract']),
        'keywords': keywords
    }

# Step 6: Multiple paper comparison and visualization
def compare_papers(paper_titles):
    paper_results = []
    
    for title in paper_titles:
        result = summarize_paper(title)
        paper_results.append(result)

    green_start = "\033[92m"
    reset_color = "\033[0m"
    magenta_color = "\033[35m"
    yellow_start = "\033[93m"

    # Print summaries and sentiment results
    for result in paper_results:
        print(f"Title: {result['title']}")
        print(f"{green_start}Summary: {result['summary']}")
        print(f"{ magenta_color}Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print(f"Abstract Length: {result['abstract_length']} words")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"{yellow_start}URL: {result['url']}\n")

    # Visualize sentiment, abstract length, and keyword comparison
    visualize_comparison(paper_results)

    # Identify the best paper for research based on key metrics
    best_paper = identify_best_paper(paper_results)
    print(f"\033[92mBest paper for research: {best_paper['title']}\033[0m")

    # Generate Venn diagram comparing keywords
    generate_venn_diagram(paper_results)

# Step 7: Visualization of key findings and sentiment comparison
def visualize_comparison(paper_results):
    titles = [result['title'] for result in paper_results]
    sentiments = [result['confidence'] for result in paper_results]
    abstract_lengths = [result['abstract_length'] for result in paper_results]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Bar chart for sentiment confidence
    ax[0].barh(titles, sentiments, color=['green' if result['sentiment'] == 'POSITIVE' else 'red' for result in paper_results])
    ax[0].set_xlabel('Sentiment Confidence')
    ax[0].set_ylabel('Research Papers')
    ax[0].set_title('Sentiment Comparison Between Papers')

    # Bar chart for abstract length
    ax[1].barh(titles, abstract_lengths, color='blue')
    ax[1].set_xlabel('Abstract Length (words)')
    ax[1].set_ylabel('Research Papers')
    ax[1].set_title('Abstract Length Comparison')

    plt.tight_layout()
    plt.show()

    # Generate word clouds for keywords
    generate_wordcloud(paper_results)

# Step 8: Generate WordCloud for keywords
def generate_wordcloud(paper_results):
    all_keywords = defaultdict(int)

    for result in paper_results:
        for keyword in result['keywords']:
            all_keywords[keyword.lower()] += 1

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(all_keywords)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Keyword Cloud Comparison')
    plt.show()

# Step 9: Generate Venn Diagram for comparing keywords between papers
def generate_venn_diagram(paper_results):
    if len(paper_results) == 2:
        keywords1 = set(paper_results[0]['keywords'])
        keywords2 = set(paper_results[1]['keywords'])

        # Generate a Venn diagram comparing keywords
        venn2([keywords1, keywords2], set_labels=(paper_results[0]['title'], paper_results[1]['title']))
        plt.title('Keyword Overlap Between Papers')
        plt.show()

    elif len(paper_results) == 3:
        keywords1 = set(paper_results[0]['keywords'])
        keywords2 = set(paper_results[1]['keywords'])
        keywords3 = set(paper_results[2]['keywords'])

        # Generate a Venn diagram for three papers
        venn3([keywords1, keywords2, keywords3], set_labels=(paper_results[0]['title'], paper_results[1]['title'], paper_results[2]['title']))
        plt.title('Keyword Overlap Between Three Papers')
        plt.show()

    else:
        print("Venn diagram only works for 2 or 3 papers at a time.")

# Step 10: Identify the best paper for research based on key metrics
def identify_best_paper(paper_results):
    best_paper = max(paper_results, key=lambda paper: (paper['confidence'], paper['abstract_length']))
    return best_paper

# Main execution for paper comparison
if __name__ == "__main__":
    # Prompt user for multiple paper titles
    paper_titles = input("Enter paper titles (comma-separated) to summarize and compare: ").split(',')

    # Trim any leading or trailing whitespace from the paper titles
    paper_titles = [title.strip() for title in paper_titles]

    # Generate comparison and visualization for the given paper titles
    compare_papers(paper_titles)
