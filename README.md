# News Mentions and Sentiment Analysis

## Project Overview
This project extracts public persons mentioned in news articles from the news website URL users input , analyzes sentiment (positive/negative) towards them, and provides visual insights using Python and Small Language Models.

The main steps include:
1. Feeding news data to the models.
2. Performing Named Entity Recognition (NER) to identify public persons.
3. Conducting sentiment analysis on sentences mentioning each person.
4. Aggregating mention counts and average sentiment.
5. Visualizing the mentioned people and their sentiment.

---

## Features / Insights
- **Single Article or Multiple Articles**
- **Mentioned People** 
- **Positive and Negative Sentiment Visualizations** 

## System Components and Technical Solution Explained

1. Data Extraction Module

This module is responsible for fetching raw text data from news articles. It acts as the system's data source.

Component: newspaper3k library.

Functionality:

Single Article Extraction: The get_single_article_data(url) function takes a single URL, downloads the article's HTML, parses it, and returns the title and main body text.

Multiple Articles Extraction: The get_multiple_articles_data(domain_url, max_articles) function builds a "source" from a domain URL, finds multiple articles linked on the front page or sitemap, and then downloads and parses each one up to the specified maximum limit.

Technical Solution:

Utilize try-except blocks to handle potential errors during network requests or parsing (e.g., article not found, parsing failure).

The @st.cache_data decorator is crucial for this component. It memoizes the results of these functions, preventing redundant web scraping and processing when a user re-runs the app with the same inputs, significantly improving performance.

2. Analysis & Processing Module
This is the core of the system where the raw text is transformed into structured, analyzable data.

Components: spaCy and Hugging Face Transformers.

Functionality:

Named Entity Recognition (NER): The spaCy library is used to identify "PERSON" entities within the article text. This step is critical as it isolates the individuals to be analyzed.

Sentiment Analysis: A pre-trained distilbert-base-uncased-finetuned-sst-2-english model from the Hugging Face Transformers library is used to analyze the sentiment of sentences containing the identified person entities. The model outputs a label (POSITIVE or NEGATIVE) and a score. The score is adjusted to be negative if the label is negative, creating a consistent numerical representation of sentiment.

Data Aggregation and Normalization: After processing each article, the extracted person data (name, sentiment score, sentiment label) is collected into a single pandas DataFrame. This DataFrame is then normalized to handle variations in names (e.g., "John Smith" and "Smith").

Technical Solutions & Algorithms:

process_text_for_entities_and_sentiment(text): This is the main algorithm.

The input text is processed by spaCy to create a Doc object.

A list of unique sentences is created to avoid duplicate analysis of the same sentence.

The algorithm iterates through all entities identified by spaCy.

If an entity's label is "PERSON," it finds the sentence containing that entity.

The sentence is then passed to the sentiment analysis model.

The sentiment score is stored, and if the label is "NEGATIVE," the score is inverted.

A dictionary with the person's name and sentiment data is created and appended to a list.

Name Normalization Algorithm:

Split Names: The split_name() function splits full names into first and last name parts. This is a simple but effective heuristic for handling different name formats.

Create Canonical Map: The create_canonical_map() function groups the DataFrame by last name. For each group, it identifies the most frequently occurring full name (e.g., if "John Smith" appears 5 times and "Smith" appears 3, "John Smith" becomes the canonical name for "Smith").

Normalize Names: The normalize_name() function applies this map. If an entry only has a last name (e.g., "Smith"), it looks up the canonical name ("John Smith") and replaces it. This helps consolidate mentions of the same individual, regardless of whether their full name was used.

Caching: The @st.cache_data and @st.cache_resource decorators are heavily used to cache the loaded NLP models and the results of data processing functions. This is crucial for performance, as model loading and processing large texts are computationally intensive tasks.

3. Presentation & Visualization Module
This module handles the display of the analyzed data to the user through the Streamlit web interface.

Component: Streamlit.

Functionality:

User Interface: Provides input fields for URLs and a slider for the number of articles, along with a radio button to switch between analysis modes.

Data Display: Uses st.dataframe() to show the raw, granular data for each mention of a person and st.bar_chart() to visualize the average sentiment score per person.

Technical Solution:

State Management: The use of st.session_state is a key feature. It stores the resulting DataFrame (results_df) across user interactions, allowing the results to persist and be displayed even after the initial analysis button is clicked. This prevents the need to re-run the entire analysis pipeline on every page reload.

Data Aggregation for Summary: A groupby() operation on the DataFrame is used to compute the average sentiment score and the total number of mentions for each person, providing a clear summary. The sort_values() method is then used to present the most mentioned individuals first.

## Files in this Repository
- `app.py` – Streamlit dashboard displaying top mentions and sentiment charts.
- `requirements.txt` – Python dependencies for running the project.

---


## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/news-extraction-app.git
cd news-extraction-app


pip install -r requirements.txt

python -m spacy download en_core_web_sm

streamlit run app.py

```
## Screenshot



Author

Name: Moe Pyae Pyae Kyaw

University: Mae Fah Luang University

License: “Permission is granted to use, modify, and distribute this software for non-commercial purposes only. Commercial use, including sale or incorporation into proprietary software, is prohibited.”

