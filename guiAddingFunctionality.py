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
import customtkinter as ctk
from matplotlib_venn import venn2, venn3

# Load environment variables from .env file
load_dotenv()

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define functions for paper analysis
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

def track_paper_trends(topic, start_year, end_year):
    yearly_summaries = {}
    yearly_keywords = defaultdict(list)
    
    for year in range(start_year, end_year + 1):
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

def summarize_text(text, model_name="facebook/bart-large-cnn", max_len=300, min_len=100):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(summary_text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(summary_text)
    return sentiment[0]

def extract_keywords(text):
    words = text.split()
    keywords = [word for word in words if word.lower() not in stop_words and len(word) > 3]
    return keywords

def summarize_paper(paper_title):
    paper_info = search_paper(paper_title)
    if paper_info is None:
        return None

    summary = summarize_text(paper_info['abstract'])
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

def compare_papers(paper_titles):
    paper_results = []
    
    for title in paper_titles:
        result = summarize_paper(title)
        if result:  # Ensure the result is not None
            paper_results.append(result)
    
    return paper_results

def ai_chatbot(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    return response['choices'][0]['message']['content']

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

def visualize_trend_keywords(yearly_keywords):
    for year, keywords in yearly_keywords.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Keyword Cloud for {year}')
        plt.show()

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

def generate_venn_diagram(paper_results):
    if len(paper_results) == 2:
        keywords1 = set(paper_results[0]['keywords'])
        keywords2 = set(paper_results[1]['keywords'])

        venn2([keywords1, keywords2], set_labels=(paper_results[0]['title'], paper_results[1]['title']))
        plt.title('Keyword Overlap Between Papers')
        plt.show()

    elif len(paper_results) == 3:
        keywords1 = set(paper_results[0]['keywords'])
        keywords2 = set(paper_results[1]['keywords'])
        keywords3 = set(paper_results[2]['keywords'])

        venn3([keywords1, keywords2, keywords3], set_labels=(paper_results[0]['title'], paper_results[1]['title'], paper_results[2]['title']))
        plt.title('Keyword Overlap Between Three Papers')
        plt.show()
    else:
        print("Venn diagram only works for 2 or 3 papers at a time.")

def identify_best_paper(paper_results):
    if not paper_results:
        return None
    best_paper = max(paper_results, key=lambda paper: (paper['confidence'], paper['abstract_length']))
    return best_paper

# GUI setup
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ResearchApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Research Summarizer and More")

        self.frame = ctk.CTkFrame(master)
        self.frame.pack(padx=20, pady=20)

        self.title_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter paper title(s), comma-separated")
        self.title_entry.pack(pady=10)

        self.year_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter year for trend analysis")
        self.year_entry.pack(pady=10)

        self.topic_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter topic for trend analysis")
        self.topic_entry.pack(pady=10)

        self.summarize_button = ctk.CTkButton(self.frame, text="Summarize Paper", command=self.summarize_paper)
        self.summarize_button.pack(pady=10)

        self.compare_button = ctk.CTkButton(self.frame, text="Compare Papers", command=self.compare_papers)
        self.compare_button.pack(pady=10)

        self.trend_button = ctk.CTkButton(self.frame, text="Track Trends", command=self.track_trends)
        self.trend_button.pack(pady=10)

        self.visualize_trend_button = ctk.CTkButton(self.frame, text="Visualize Trends", command=self.visualize_trends)
        self.visualize_trend_button.pack(pady=10)

        self.visualize_comparison_button = ctk.CTkButton(self.frame, text="Visualize Comparison", command=self.visualize_comparison)
        self.visualize_comparison_button.pack(pady=10)

        self.wordcloud_button = ctk.CTkButton(self.frame, text="Generate Word Cloud", command=self.generate_wordcloud)
        self.wordcloud_button.pack(pady=10)

        self.venn_button = ctk.CTkButton(self.frame, text="Generate Venn Diagram", command=self.generate_venn_diagram)
        self.venn_button.pack(pady=10)

        self.best_paper_button = ctk.CTkButton(self.frame, text="Identify Best Paper", command=self.identify_best_paper)
        self.best_paper_button.pack(pady=10)

        self.chat_button = ctk.CTkButton(self.frame, text="AI Chatbot", command=self.ai_chatbot_interface)
        self.chat_button.pack(pady=10)

        self.output_text = ctk.CTkTextbox(self.frame, height=15, width=60)
        self.output_text.pack(pady=10)

        self.results = []

    def summarize_paper(self):
        paper_title = self.title_entry.get()
        result = summarize_paper(paper_title)
        self.display_result(result)

    def compare_papers(self):
        paper_titles = self.title_entry.get().split(',')
        paper_titles = [title.strip() for title in paper_titles]
        self.results = compare_papers(paper_titles)
        self.display_comparison(self.results)

    def track_trends(self):
        topic = self.topic_entry.get()
        start_year = int(self.year_entry.get().split('-')[0])
        end_year = int(self.year_entry.get().split('-')[1])
        yearly_summaries, yearly_keywords = track_paper_trends(topic, start_year, end_year)
        self.output_text.insert('end', f"Trend tracking results for {topic} from {start_year} to {end_year}\n")
        self.visualize_trends(yearly_summaries, yearly_keywords)

    def visualize_trends(self, yearly_summaries, yearly_keywords):
        visualize_paper_trends(yearly_summaries)
        visualize_trend_keywords(yearly_keywords)

    def visualize_comparison(self):
        if not self.results:
            self.output_text.insert('end', "No comparison results available.\n")
            return
        visualize_comparison(self.results)
        generate_wordcloud(self.results)
        generate_venn_diagram(self.results)

    def generate_wordcloud(self):
        if not self.results:
            self.output_text.insert('end', "No papers available for word cloud.\n")
            return
        generate_wordcloud(self.results)

    def generate_venn_diagram(self):
        if not self.results:
            self.output_text.insert('end', "No papers available for Venn diagram.\n")
            return
        generate_venn_diagram(self.results)

    def identify_best_paper(self):
        best_paper = identify_best_paper(self.results)
        if best_paper:
            self.output_text.insert('end', f"Best Paper: {best_paper['title']} with confidence {best_paper['confidence']}\n")
        else:
            self.output_text.insert('end', "No papers to evaluate.\n")

    def ai_chatbot_interface(self):
        query_window = ctk.CTkToplevel(self.master)
        query_window.title("AI Chatbot")

        query_entry = ctk.CTkEntry(query_window, placeholder_text="Ask the AI chatbot")
        query_entry.pack(pady=10)

        submit_button = ctk.CTkButton(query_window, text="Submit", command=lambda: self.chatbot_query(query_entry.get()))
        submit_button.pack(pady=10)

        self.chat_output = ctk.CTkTextbox(query_window, height=10, width=50)
        self.chat_output.pack(pady=10)

    def display_result(self, result):
        if result:
            self.output_text.insert('end', f"Title: {result['title']}\nSummary: {result['summary']}\nSentiment: {result['sentiment']} (Confidence: {result['confidence']})\nURL: {result['url']}\n\n")
        else:
            self.output_text.insert('end', "Paper not found.\n")

    def display_comparison(self, results):
        if not results:
            self.output_text.insert('end', "No papers found for comparison.\n")
            return
        
        for result in results:
            self.output_text.insert('end', f"Title: {result['title']}\nSummary: {result['summary']}\nSentiment: {result['sentiment']} (Confidence: {result['confidence']})\n\n")

    def chatbot_query(self, user_query):
        answer = ai_chatbot(user_query)
        self.chat_output.insert('end', f"Chatbot: {answer}\n")

# Main execution
if __name__ == "__main__":
    root = ctk.CTk()
    app = ResearchApp(root)
    root.mainloop()
