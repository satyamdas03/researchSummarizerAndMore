import requests
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import openai
from dotenv import load_dotenv
import os
from matplotlib_venn import venn2, venn3

# Define color codes for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Load environment variables from .env file
load_dotenv()

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure your .env has this variable

# Function to initiate user mode selection
def select_mode():
    while True:
        mode = input("Do you want to use the AI chatbot for research assistance or the manual summarizer? (Type 'AI' or 'Manual'): ").strip().lower()
        if mode in ['ai', 'manual']:
            return mode
        print("Invalid choice. Please type 'AI' or 'Manual'.")

# Step 1: Fetch paper details from arXiv
def search_paper(paper_title):
    query = f"ti:{paper_title}"
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query={query}&start=0&max_results=1"
    response = requests.get(base_url + search_query)
    
    if response.status_code == 200:
        return parse_arxiv_response(response.text)
    else:
        return None

def parse_arxiv_response(response):
    root = ET.fromstring(response)
    entry = root.find('{http://www.w3.org/2005/Atom}entry')
    
    if entry is not None:
        return {
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
            'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
            'url': entry.find('{http://www.w3.org/2005/Atom}id').text
        }
    return None

# New searching function for arXiv
def search_papers_by_topic_and_year(topic, year):
    query = f"all:{topic}+AND+submittedDate:[{year}0101 TO {year}1231]"
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query={query}&start=0&max_results=10"
    response = requests.get(base_url + search_query)
    
    papers = []
    
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            papers.append({
                'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                'abstract': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                'year': year,
                'url': entry.find('{http://www.w3.org/2005/Atom}id').text
            })
    return papers

# Function to fetch papers for a range of years and track trends
def track_paper_trends(topic, start_year, end_year):
    yearly_summaries = {}
    yearly_keywords = defaultdict(list)
    
    for year in range(start_year, end_year + 1):
        print(f"Fetching papers for the year {year}...")
        papers = search_papers_by_topic_and_year(topic, year)
        summaries = []
        keywords_list = []
        
        for paper in papers:
            summary = summarize_text(paper['abstract'])
            keywords = extract_keywords(paper['abstract'])
            summaries.append(summary)
            keywords_list.extend(keywords)
        
        yearly_summaries[year] = summaries
        yearly_keywords[year] = keywords_list
    
    return yearly_summaries, yearly_keywords

# Visualization: Number of papers per year
def visualize_paper_trends(yearly_summaries):
    years = list(yearly_summaries.keys())
    num_papers = [len(summaries) for summaries in yearly_summaries.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(years, num_papers, marker='o', linestyle='-', color='b')
    plt.fill_between(years, num_papers, color='lightblue', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.title('Research Paper Trends Over Time')
    plt.grid(True)
    plt.show()

# Visualization: Word cloud of emerging themes by year
def visualize_trend_keywords(yearly_keywords):
    for year, keywords in yearly_keywords.items():
        print(f"Generating word cloud for the year {year}...")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Keyword Cloud for {year}')
        plt.show()

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
    if paper_info is None:
        return None

    summary = summarize_text(paper_info['abstract'], max_len=max_len, min_len=min_len)
    sentiment_result = analyze_sentiment(summary)
    keywords = extract_keywords(paper_info['abstract'])
    
    return {
        'title': paper_info['title'],
        'summary': summary,
        'url': paper_info['url'],
        'sentiment': sentiment_result['label'],
        'confidence': sentiment_result['score'],
        'abstract_length': len(paper_info['abstract']),
        'keywords': keywords
    }

# Step 6: Multiple paper comparison and visualization
def compare_papers(paper_titles):
    paper_results = []
    
    for title in paper_titles:
        result = summarize_paper(title)
        if result:  # Ensure the result is not None
            paper_results.append(result)

    green_start = "\033[92m"
    reset_color = "\033[0m"
    magenta_color = "\033[35m"
    yellow_start = "\033[93m"

    for result in paper_results:
        print(f"Title: {result['title']}")
        print(f"{green_start}Summary: {result['summary']}{reset_color}")

        print(f"{magenta_color}Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f}){reset_color}")
        print(f"Abstract Length: {result['abstract_length']} words")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print(f"{yellow_start}URL: {result['url']}{reset_color}\n")

    # Visualize sentiment, abstract length, and keyword comparison
    visualize_comparison(paper_results)

    # Identify the best paper for research based on key metrics
    best_paper = identify_best_paper(paper_results)
    if best_paper:
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
    if not paper_results:
        return None
    best_paper = max(paper_results, key=lambda paper: (paper['confidence'], paper['abstract_length']))
    return best_paper

# Step 11: AI Chatbot for Research Assistance
def ai_chatbot(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    return response['choices'][0]['message']['content']

# Main execution for paper comparison
# if __name__ == "__main__":
#     # Prompt user for multiple paper titles
#     paper_titles = input("Enter paper titles (comma-separated) to summarize and compare: ").split(',')

#     # Trim any leading or trailing whitespace from the paper titles
#     paper_titles = [title.strip() for title in paper_titles]

#     # Generate comparison and visualization for the given paper titles
#     compare_papers(paper_titles)

#     # Prompt user for the topic and the range of years
#     topic = input("Enter a topic to track research trends: ")
#     start_year = int(input("Enter the start year: "))
#     end_year = int(input("Enter the end year: "))
    
#     # Track paper trends and visualize
#     yearly_summaries, yearly_keywords = track_paper_trends(topic, start_year, end_year)
    
#     # Visualize the trends
#     visualize_paper_trends(yearly_summaries)
#     visualize_trend_keywords(yearly_keywords)

#     # Chatbot interaction
#     while True:
#         user_query = input("Ask the AI chatbot for research assistance (type 'exit' to quit): ")
#         if user_query.lower() == 'exit':
#             break
#         answer = ai_chatbot(user_query)
#         print(f"Chatbot: {answer}")

# Main execution for mode selection
if __name__ == "__main__":
    # Select mode
    mode = select_mode()

    if mode == 'manual':
        # Manual mode
        paper_titles = input("Enter paper titles (comma-separated) to summarize and compare: ").split(',')
        paper_titles = [title.strip() for title in paper_titles]
        compare_papers(paper_titles)

        topic = input("Enter a topic to track research trends: ")
        start_year = int(input("Enter the start year: "))
        end_year = int(input("Enter the end year: "))
        
        yearly_summaries, yearly_keywords = track_paper_trends(topic, start_year, end_year)
        
        visualize_paper_trends(yearly_summaries)
        visualize_trend_keywords(yearly_keywords)

    else:
        # AI chatbot mode
        while True:
            user_query = input("Ask the AI chatbot for research assistance (type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
            answer = ai_chatbot(user_query)
            print(f"{BLUE}User: {user_query}{RESET}")
            print(f"{GREEN}Chatbot: {answer}{RESET}")
