from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import nltk
from textblob import TextBlob
from googletrans import Translator

app = Flask(__name__)

df = pd.read_csv('./dataset.csv')
df_verified = df[df['verified_purchase'] == True]

translator = Translator()

def analyze_sentiment(reviews):
    sentiments = []
    for review in reviews:
        translated_review = translator.translate(review, dest='en').text
        sentiment = TextBlob(translated_review).sentiment.polarity
        sentiments.append(sentiment)
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return average_sentiment

def analyze_product(product_name):
    print("Analyzing product:", product_name)
    df_product = df_verified[df_verified['product_name'] == product_name]
    if df_product.empty:
        print("No reviews found for this product:", product_name)
        return None, "No reviews found for this product."
    average_rating = df_product['review_rating'].mean()
    print("Average Rating:", average_rating)
    rating_counts = df_product['review_rating'].value_counts().sort_index()
    print("Rating Counts:", rating_counts)
    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.title(f'Number of Reviews per Star Rating for {product_name}')
    plt.xticks(range(1, 6))
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    review_texts = df_product['review_text'].dropna().tolist()
    average_sentiment = analyze_sentiment(review_texts)
    sentiment_label = "Positive" if average_sentiment > 0 else "Negative" if average_sentiment < 0 else "Neutral"
    print("Average Sentiment:", average_sentiment, "Sentiment Label:", sentiment_label)
    recommendation = "Buy" if average_rating >= 3 and sentiment_label == "Positive" else "Don't Buy"
    print("Recommendation:", recommendation)
    result = (
        f"Average Rating: {average_rating:.2f}. "
        f"Average Sentiment: {average_sentiment:.2f} ({sentiment_label}). "
        f"Recommendation: {recommendation}"
    )
    return plot_url, result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_name = request.form['product_name']
        plot_url, result = analyze_product(product_name)
        return render_template('index.html', plot_url=plot_url, result=result, product_name=product_name)
    return render_template('index.html')

@app.route('/product_names')
def product_names():
    product_names = df_verified['product_name'].unique().tolist()
    return {"product_names": product_names}

if __name__ == '__main__':
    nltk.download('punkt')
    app.run(port=8000, debug=True)
