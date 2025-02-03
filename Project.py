# ======================================
#    Import Libraries
# ======================================
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nrclex import NRCLex

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

# ======================================
#    Helper Functions
# ======================================

def preprocess_text(text):
    """
    Clean and preprocess tweet text.
    Steps:
    1. Lowercase text
    2. Remove URLs, mentions, hashtags, RT, numbers, punctuation
    3. Remove non-ASCII characters and noisy patterns
    4. Remove stopwords (English + Multilingual)
    5. Lemmatize words
    """
    if pd.isna(text):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove noise (URLs, mentions, hashtags, RT, numbers, punctuation)
    text = re.sub(r'http\S+|@\w+|#\w+|\brt\b|\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove non-ASCII characters and noisy patterns
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\w*diadia\w*', '', text)  # Remove specific noise patterns
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    for lang in ['french', 'italian', 'spanish', 'german']:
        stop_words.update(stopwords.words(lang))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens).strip()


def analyze_sentiment(text, analyzer):
    """
    Return sentiment label (positive, negative, neutral)
    using VADER SentimentIntensityAnalyzer.
    """
    if pd.isna(text) or text.strip() == '':
        return 'neutral'
    scores = analyzer.polarity_scores(text)
    return 'positive' if scores['compound'] >= 0.05 else 'negative' if scores['compound'] <= -0.05 else 'neutral'


def get_textblob_sentiment(text):
    """
    Return sentiment from TextBlob polarity score.
    """
    if not text.strip():
        return 'neutral'
    polarity = TextBlob(text).sentiment.polarity
    return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'


def get_emotion(text):
    """
    Return dominant emotion using NRCLex.
    """
    if pd.isna(text) or text.strip() == '':
        return 'neutral'
    emotion_text = NRCLex(text)
    return max(emotion_text.top_emotions, key=lambda x: x[1])[0] if emotion_text.top_emotions else 'neutral'


def generate_wordcloud(text, title):
    """
    Generate and display a WordCloud for given text.
    """
    if not text.strip():
        print(f"No text available for {title}")
        return
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=stopwords.words('english')
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14)
    plt.show()


# ======================================
#    1. Load Dataset
# ======================================
file_path = '30K Tweets with russiaukrainewar hashtag.csv'
df = pd.read_csv(file_path)

# ======================================
#    2. Preprocessing
# ======================================
df['Refined_Tweet'] = df['Tweet'].apply(preprocess_text)

# Quick Validation
print(df[['Tweet', 'Refined_Tweet']].head())

# ======================================
#    3. Sentiment Analysis
# ======================================
vader_analyzer = SentimentIntensityAnalyzer()

df['TextBlob_Sentiment'] = df['Refined_Tweet'].apply(get_textblob_sentiment)
df['VADER_Sentiment'] = df['Refined_Tweet'].apply(lambda x: analyze_sentiment(x, vader_analyzer))

# Plot Sentiment Distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x='TextBlob_Sentiment', data=df, ax=axs[0])
sns.countplot(x='VADER_Sentiment', data=df, ax=axs[1])
axs[0].set_title('TextBlob Sentiment')
axs[1].set_title('VADER Sentiment')
plt.tight_layout()
plt.show()

# ======================================
#    4. Emotion Analysis
# ======================================
df['Emotion'] = df['Refined_Tweet'].apply(get_emotion)

# Plot Emotion Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Emotion', data=df, order=df['Emotion'].value_counts().index, palette='coolwarm')
plt.title('Emotion Distribution')
plt.show()

# ======================================
#    5. WordClouds by Sentiment
# ======================================
for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_text = ' '.join(df[df['TextBlob_Sentiment'] == sentiment]['Refined_Tweet'])
    if sentiment_text.strip():
        generate_wordcloud(sentiment_text, f'WordCloud for {sentiment.capitalize()} Sentiment')

# ======================================
#    6. WordClouds by Emotion
# ======================================
unique_emotions = df['Emotion'].unique()
for emotion in unique_emotions:
    emotion_text = ' '.join(df[df['Emotion'] == emotion]['Refined_Tweet'])
    if emotion_text.strip():
        generate_wordcloud(emotion_text, f'WordCloud for {emotion.capitalize()} Emotion')

# ======================================
#    Emotion Over Time Analysis
# ======================================

# Ensure the 'Time' column is in datetime format and extract the date
df['Time'] = pd.to_datetime(df['Time'])
df['Date'] = df['Time'].dt.date

# Group data by Date and Emotion
emotion_over_time = df.groupby(['Date', 'Emotion']).size().unstack(fill_value=0)

# Find the peak date for each emotion
emotion_peaks = emotion_over_time.idxmax()
peak_values = emotion_over_time.max()

# Print the peak dates and values for each emotion
print("Emotion Peaks:")
for emotion, peak_date in emotion_peaks.items():
    print(f"- {emotion.capitalize()}: {peak_date} ({peak_values[emotion]} tweets)")

# Plot emotion trends over time
plt.figure(figsize=(12, 8))
ax = plt.gca()
emotion_over_time.plot(ax=ax, marker='o', linewidth=2)

# Highlight peak times with vertical red lines
for emotion, peak_date in emotion_peaks.items():
    ax.axvline(pd.Timestamp(peak_date), color='red', linestyle='--', alpha=0.7)

# Add title, labels, and legend
plt.title("Emotion Trends Over Time with Peaks Highlighted")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================
#    8. Save Final Dataset
# ======================================
output_path = 'refined_tweets_with_sentiments_emotions.csv'
df.to_csv(output_path, index=False)
print(f"✅ Dataset saved as {output_path}")

