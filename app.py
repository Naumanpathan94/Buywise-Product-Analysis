from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import asyncio
import nest_asyncio
import pandas as pd
import nltk
from googletrans import Translator

nest_asyncio.apply()

app = Flask(__name__)

# In-memory storage for user reviews per product
user_reviews_db = {}

# Load your dataset once at the top (adjust the path as needed)
df = pd.read_csv('./dataset.csv')
df_verified = df[df['verified_purchase'] == True]

translator = Translator()

async def async_translate(translator, text, dest='en'):
    translation = await translator.translate(text, dest=dest)
    return translation.text

def analyze_sentiment(review_texts):
    sentiments = []

    async def gather_translations():
        tasks = [async_translate(translator, review, dest='en') for review in review_texts]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            import threading
            result = []
            def run():
                result.append(loop.run_until_complete(gather_translations()))
            t = threading.Thread(target=run)
            t.start()
            t.join()
            translated_reviews = result[0]
        else:
            translated_reviews = loop.run_until_complete(gather_translations())
    except RuntimeError:
        translated_reviews = asyncio.run(gather_translations())

    sentiments = [TextBlob(text).sentiment.polarity for text in translated_reviews]
    return sentiments

def generate_bar_chart(ratings, sentiments):
    plt.figure(figsize=(6, 4))
    plt.bar(['1', '2', '3', '4', '5'], [ratings.count(i) for i in range(1, 6)], color='#60a5fa', alpha=0.7, label='Ratings')
    plt.twinx()
    plt.plot(range(1, len(sentiments)+1), sentiments, color='#6366f1', marker='o', label='Sentiment')
    plt.ylabel('Sentiment Score')
    plt.title('Rating Distribution & Sentiment Trend')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# Print your DataFrame columns to debug the actual column names
print("CSV Columns:", df.columns.tolist())

def get_product_reviews_and_ratings(product_name):
    """
    Returns a list of (review_text, rating) tuples for the given product_name.
    Adjusted to match your dataset.csv columns:
    ['url', 'product_name', 'reviewer_name', 'review_title', 'review_text', 'review_rating', 'verified_purchase', 'review_date', 'helpful_count', 'uniq_id', 'scraped_at']
    """
    review_col = 'review_text'
    rating_col = 'review_rating'
    product_col = 'product_name'

    # Defensive check (optional, can be removed after confirming it works)
    for col in [review_col, rating_col, product_col]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )

    product_rows = df[df[product_col] == product_name]
    reviews_and_ratings = list(zip(product_rows[review_col].astype(str), product_rows[rating_col].astype(int)))
    return reviews_and_ratings

def analyze_product(product_name):
    # Get original reviews and ratings
    reviews_and_ratings = get_product_reviews_and_ratings(product_name)
    review_texts = [r[0] for r in reviews_and_ratings]
    ratings = [r[1] for r in reviews_and_ratings]

    # Add user reviews if any
    user_reviews = user_reviews_db.get(product_name, [])
    review_texts += [r['text'] for r in user_reviews]
    ratings += [r['rating'] for r in user_reviews]

    # Sentiment analysis
    sentiments = analyze_sentiment(review_texts)
    avg_sentiment = round(sum(sentiments) / len(sentiments), 3) if sentiments else 0
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0

    # Recommendation: Only "Buy" or "Don't Buy"
    if avg_sentiment > 0.1 and avg_rating >= 3.5:
        recommendation = "Buy"
    else:
        recommendation = "Don't Buy"

    plot_url = generate_bar_chart(ratings, sentiments)
    result = f"Average Rating: {avg_rating} | Average Sentiment: {avg_sentiment} | Recommendation: {recommendation}"
    return plot_url, result, avg_rating, avg_sentiment, recommendation

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    plot_url = None
    product_name = None
    user_review = None
    user_sentiment_score = None
    user_sentiment_category = None
    avg_rating = None
    avg_sentiment = None
    recommendation = None

    if request.method == 'POST':
        if request.form.get('submit_review'):
            user_review = request.form.get('user_review', '')
            product_name = request.form.get('product_name', '')
            if user_review and product_name:
                # Translate user review to English for sentiment analysis
                try:
                    translated = translator.translate(user_review, dest='en').text
                except Exception:
                    translated = user_review  # fallback if translation fails
                blob = TextBlob(translated)
                user_sentiment_score = round(blob.sentiment.polarity, 3)
                if user_sentiment_score > 0.1:
                    user_sentiment_category = 'positive'
                elif user_sentiment_score < -0.1:
                    user_sentiment_category = 'negative'
                else:
                    user_sentiment_category = 'neutral'
                # Save user review with dummy rating (e.g., 3 for neutral, 5 for positive, 1 for negative)
                if user_sentiment_category == 'positive':
                    user_rating = 5
                elif user_sentiment_category == 'negative':
                    user_rating = 1
                else:
                    user_rating = 3
                user_reviews_db.setdefault(product_name, []).append({'text': user_review, 'rating': user_rating})
                # Recalculate product stats (includes user review)
                plot_url, result, avg_rating, avg_sentiment, recommendation = analyze_product(product_name)
                # Ensure user sentiment is always passed to template after review submission
                return render_template(
                    'index.html',
                    result=result,
                    plot_url=plot_url,
                    product_name=product_name,
                    user_review=user_review,
                    user_sentiment_score=user_sentiment_score,
                    user_sentiment_category=user_sentiment_category,
                    avg_rating=avg_rating,
                    avg_sentiment=avg_sentiment,
                    recommendation=recommendation,
                )
        product_name = request.form.get('product_name', '')
        if product_name:
            plot_url, result, avg_rating, avg_sentiment, recommendation = analyze_product(product_name)
            # Show the latest user review sentiment if available
            user_reviews = user_reviews_db.get(product_name, [])
            if user_reviews:
                last_user_review = user_reviews[-1]['text']
                try:
                    translated = translator.translate(last_user_review, dest='en').text
                except Exception:
                    translated = last_user_review
                blob = TextBlob(translated)
                user_sentiment_score = round(blob.sentiment.polarity, 3)
                if user_sentiment_score > 0.1:
                    user_sentiment_category = 'positive'
                elif user_sentiment_score < -0.1:
                    user_sentiment_category = 'negative'
                else:
                    user_sentiment_category = 'neutral'
                user_review = last_user_review
            else:
                user_review = None
                user_sentiment_score = None
                user_sentiment_category = None
        return render_template(
            'index.html',
            result=result,
            plot_url=plot_url,
            product_name=product_name,
            user_review=user_review,
            user_sentiment_score=user_sentiment_score,
            user_sentiment_category=user_sentiment_category,
            avg_rating=avg_rating,
            avg_sentiment=avg_sentiment,
            recommendation=recommendation,
        )
    return render_template(
        'index.html',
        result=result,
        plot_url=plot_url,
        product_name=product_name,
        user_review=user_review,
        user_sentiment_score=user_sentiment_score,
        user_sentiment_category=user_sentiment_category,
        avg_rating=avg_rating,
        avg_sentiment=avg_sentiment,
        recommendation=recommendation,
    )

@app.route('/product_names')
def product_names():
    product_names = df_verified['product_name'].unique().tolist()
    return {"product_names": product_names}

@app.route('/submit_review', methods=['POST'])
def submit_review():
    user_review = request.form.get('user_review', '')
    product_name = request.form.get('product_name', '')
    user_sentiment_score = None
    user_sentiment_category = None
    avg_rating = None
    avg_sentiment = None
    recommendation = None
    plot_url = None
    result = None

    if user_review and product_name:
        # Translate user review to English for sentiment analysis
        try:
            translated = translator.translate(user_review, dest='en').text
        except Exception:
            translated = user_review  # fallback if translation fails
        blob = TextBlob(translated)
        user_sentiment_score = round(blob.sentiment.polarity, 3)
        if user_sentiment_score > 0.1:
            user_sentiment_category = 'positive'
        elif user_sentiment_score < -0.1:
            user_sentiment_category = 'negative'
        else:
            user_sentiment_category = 'neutral'
        # Save user review with dummy rating (e.g., 3 for neutral, 5 for positive, 1 for negative)
        if user_sentiment_category == 'positive':
            user_rating = 5
        elif user_sentiment_category == 'negative':
            user_rating = 1
        else:
            user_rating = 3
        user_reviews_db.setdefault(product_name, []).append({'text': user_review, 'rating': user_rating})
        # Recalculate product stats (includes user review)
        plot_url, result, avg_rating, avg_sentiment, recommendation = analyze_product(product_name)
    return jsonify({
        'user_sentiment_score': user_sentiment_score,
        'user_sentiment_category': user_sentiment_category,
        'avg_rating': avg_rating,
        'avg_sentiment': avg_sentiment,
        'recommendation': recommendation,
        'plot_url': plot_url,
        'result': result
    })

if __name__ == '__main__':
    nltk.download('punkt')
    app.run(port=8000, debug=True)
