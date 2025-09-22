import streamlit as st
import spacy
import pandas as pd
from newspaper import Article, build
from transformers import pipeline


# Model Loading

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        st.stop()

@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except OSError:
        st.error("Sentiment Model not found. Please run 'pip install transformers' in your terminal.")
        st.stop()

nlp = load_spacy_model()
sentiment_model = load_sentiment_model()

#Analyzing functions for sentiment and entities

@st.cache_data
def process_text_for_entities_and_sentiment(text):
    doc = nlp(text)
    persons = []
    #To avoid duplicate sentences, process sentences once
    unique_sentences = {sent.text: sent for sent in doc.sents}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            sentence = next((s for s in unique_sentences.values() if ent.text in s.text), None)
            if sentence:
                #The sentiment model returns a list of a single dictionary, so we take the first item
                sentiment_result = sentiment_model(sentence.text)[0]
                
                #Adjust the score to be negative if the label is 'NEGATIVE'
                score = sentiment_result["score"]
                if sentiment_result["label"] == 'NEGATIVE':
                    score = -score
                
                persons.append({
                    "name": ent.text,
                    "sentiment_score": score,
                    "sentiment_label": sentiment_result["label"]
                })
    return persons

#Split name into first and last for normalization
def split_name(name):
    parts = str(name).split()
    if len(parts) == 1:
        return ("", parts[0])
    else:
        return (parts[0], parts[-1])

#Create a canonical name mapping based on the most frequent first name for each last name
def create_canonical_map(df):
   
    canonical_map = (
        df[df["first"] != ""]
        .groupby("last")["name"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    return canonical_map

#Normalize names in the dataframe using the canonical map if it has only last name
def normalize_name(row, canonical_map):
    if row["first"] == "" and row["last"] in canonical_map:
        return canonical_map[row["last"]]
    return row["name"]

#Data Extraction Functions

def get_single_article_data(url):
    """Downloads and extracts data from a single article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {"title": article.title, "text": article.text}
    except Exception as e:
        return {"error": f"Error processing single article at {url}: {e}"}

@st.cache_data
def get_multiple_articles_data(domain_url, max_articles=5):
    """Downloads and extracts data from a domain with a max limit."""
    articles_data = []
    try:
        source = build(domain_url, memoize_articles=False)
        articles_to_process = source.articles[:max_articles]

        for article in articles_to_process:
            try:
                article.download()
                article.parse()
                if article.text and article.title:
                    articles_data.append({"title": article.title, "text": article.text, "url": article.url})
            except Exception as e:
                st.warning(f"Could not process article from {article.url}: {e}")

    except Exception as e:
        return {"error": f"Error building source from {domain_url}: {e}"}

    return articles_data

#Streamlit Dashboard ---

st.title("Public Person Sentiment Analyzer")
st.write("Analyze public persons and their sentiment from news articles.")

analysis_mode = st.radio("Choose Analysis Mode:", ("Single Article", "Multiple Articles from a Domain"))

if 'results_df' not in st.session_state:
    st.session_state['results_df'] = pd.DataFrame()

if analysis_mode == "Single Article":
    st.subheader("Analyze a Single Article")
    url = st.text_input("Enter Article URL:", key="single_url")
    if st.button("Analyze Article"):
        with st.spinner("Analyzing..."):
            article_data = get_single_article_data(url)
            if "error" in article_data:
                st.error(article_data["error"])
            elif article_data["text"]:
                persons = process_text_for_entities_and_sentiment(article_data["text"])

                if persons:
                    df = pd.DataFrame(persons)
                    df[["first", "last"]] = pd.DataFrame(df["name"].apply(split_name).tolist(), index=df.index)

                    if not df[df["first"] != ""].empty:
                        canonical_map = create_canonical_map(df)
                        df["name"] = df.apply(lambda row: normalize_name(row, canonical_map), axis=1)

                    df = df.drop(columns=["first", "last"])
                    df['source'] = article_data['title']
                    st.session_state['results_df'] = df
                    st.success(f"Successfully analyzed article: {article_data['title']}")
                else:
                    st.warning("No public persons found in this article.")

            else:
                st.warning("Could not extract text from the provided URL.")

elif analysis_mode == "Multiple Articles from a Domain":
    st.subheader("Analyze Multiple Articles from a Domain")
    domain_url = st.text_input("Enter Domain URL (e.g., https://www.bbc.com):", key="domain_url")
    max_articles = st.slider("Number of articles to analyze:", 1, 20, 5)

    if st.button("Analyze Domain"):
        with st.spinner(f"Analyzing {max_articles} articles from {domain_url}..."):
            articles_data = get_multiple_articles_data(domain_url, max_articles)

            if "error" in articles_data:
                st.error(articles_data["error"])
            else:
                all_persons = []
                #First, collect all person data from all articles
                for article in articles_data:
                    persons = process_text_for_entities_and_sentiment(article['text'])
                    if persons:
                        for p in persons:
                            p['source'] = article['title']
                        all_persons.extend(persons)

                if all_persons:
                    #Create a single DataFrame from ALL collected person data
                    final_df = pd.DataFrame(all_persons)

                    #Perform the name normalization on the complete DataFrame
                    final_df[["first", "last"]] = pd.DataFrame(final_df["name"].apply(split_name).tolist(), index=final_df.index)

                    if not final_df[final_df["first"] != ""].empty:
                        canonical_map = create_canonical_map(final_df)
                        final_df["name"] = final_df.apply(lambda row: normalize_name(row, canonical_map), axis=1)

                    final_df = final_df.drop(columns=["first", "last"])
                    st.session_state['results_df'] = final_df
                    st.success(f"Successfully analyzed {len(articles_data)} articles from {domain_url}")
                else:
                    st.warning("No public persons found in these articles.")

#Display Results

if not st.session_state['results_df'].empty:
    st.markdown("---")
    st.subheader("Raw Analysis Data")
    st.dataframe(st.session_state['results_df'])

    st.subheader("Analysis Summary")
    df_summary = st.session_state['results_df'].groupby('name')['sentiment_score'].agg(['mean', 'count']).reset_index()
    df_summary.rename(columns={'mean': 'Average Sentiment Score', 'count': 'Mentions'}, inplace=True)

    st.dataframe(df_summary.sort_values(by='Mentions', ascending=False).set_index("name"))

    if not df_summary.empty:
        st.markdown("Average Sentiment Score by Person")
        st.bar_chart(df_summary, x='name', y='Average Sentiment Score')
    else:
        st.info("No data to display in the chart.")
