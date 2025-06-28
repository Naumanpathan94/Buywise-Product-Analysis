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

def generate_bar_chart(ratings, sentiments, user_reviews=None):
    # Optionally include user reviews in the bar chart
    if user_reviews and len(user_reviews) > 0:
        ratings = ratings + [r['rating'] for r in user_reviews]
        # For sentiments, we assume user reviews are already included in sentiments list
    plt.figure(figsize=(20, 18))
    plt.bar(['1', '2', '3', '4', '5'], [ratings.count(i) for i in range(1, 6)], color='#60a5fa', alpha=0.7, label='Ratings')
    plt.twinx()
    plt.plot(range(1, len(sentiments)+1), sentiments, color='#6366f1', marker='o', linewidth=3, markersize=10, label='Sentiment')
    plt.ylabel('Sentiment Score', fontsize=18)
    plt.title('Rating Distribution & Sentiment Trend', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def generate_sentiment_over_time_chart(product_rows, user_reviews):
    import datetime
    df_time = product_rows.copy()
    # Only keep rows with non-empty review_text and review_date
    if 'review_date' not in df_time.columns or 'review_text' not in df_time.columns:
        return None
    # Convert review_date to string to avoid pandas parsing issues with mixed types
    df_time['review_date'] = df_time['review_date'].astype(str)
    df_time = df_time[df_time['review_text'].notnull() & df_time['review_date'].notnull()]
    # Try to parse dates robustly
    df_time['review_date'] = pd.to_datetime(df_time['review_date'].str.extract(r'(\d{1,2} \w+ \d{4})')[0], errors='coerce')
    df_time = df_time.dropna(subset=['review_date'])
    # If user_reviews exist, merge them, else just use dataset
    if user_reviews and len(user_reviews) > 0:
        now = datetime.datetime.now()
        user_df = pd.DataFrame([{
            'review_text': r['text'],
            'review_date': now + datetime.timedelta(seconds=i),
            'verified_purchase': True
        } for i, r in enumerate(user_reviews)])
        for col in df_time.columns:
            if col not in user_df.columns:
                user_df[col] = None
        user_df = user_df[df_time.columns]
        user_df['review_date'] = pd.to_datetime(user_df['review_date'], errors='coerce')
        df_time = pd.concat([df_time, user_df], ignore_index=True)
        df_time = df_time.dropna(subset=['review_date', 'review_text'])
    if df_time.empty:
        return None
    review_texts = df_time['review_text'].astype(str).tolist()
    sentiments = analyze_sentiment(review_texts)
    df_time = df_time.iloc[:len(sentiments)].copy()
    df_time['sentiment'] = sentiments
    df_time = df_time.sort_values('review_date')
    if df_time.empty or df_time['sentiment'].isnull().all():
        return None
    plt.figure(figsize=(16, 14.4))  # 80% of (20, 18)
    plt.plot(df_time['review_date'], df_time['sentiment'], marker='o', color='#38bdf8', linewidth=3, markersize=12, label='Sentiment')
    plt.fill_between(df_time['review_date'], df_time['sentiment'], color='#38bdf8', alpha=0.15)
    plt.title('Sentiment Over Time', fontsize=26, color='#38bdf8')
    plt.xlabel('Date', fontsize=22)
    plt.ylabel('Sentiment Score', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout(pad=2)
    plt.legend(fontsize=16)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def generate_verified_pie_chart(product_rows, user_reviews):
    import datetime
    df_pie = product_rows.copy()
    if 'verified_purchase' not in df_pie.columns:
        return None
    df_pie = df_pie[df_pie['verified_purchase'].notnull()]
    # Merge user reviews as verified purchases
    if user_reviews and len(user_reviews) > 0:
        now = datetime.datetime.now()
        user_df = pd.DataFrame([{
            'review_text': r['text'],
            'review_date': now + datetime.timedelta(seconds=i),
            'verified_purchase': True
        } for i, r in enumerate(user_reviews)])
        for col in df_pie.columns:
            if col not in user_df.columns:
                user_df[col] = None
        user_df = user_df[df_pie.columns]
        user_df['verified_purchase'] = True
        df_pie = pd.concat([df_pie, user_df], ignore_index=True)
    # Only count True/False, ignore NaN
    verified_count = df_pie[df_pie['verified_purchase'] == True].shape[0]
    non_verified_count = df_pie[df_pie['verified_purchase'] == False].shape[0]
    if verified_count + non_verified_count == 0:
        return None
    labels = []
    sizes = []
    colors = []
    explode = []
    if verified_count > 0:
        labels.append('Verified')
        sizes.append(verified_count)
        colors.append('#22c55e')
        explode.append(0.05)
    if non_verified_count > 0:
        labels.append('Non-Verified')
        sizes.append(non_verified_count)
        colors.append('#ef4444')
        explode.append(0.05)
    plt.figure(figsize=(17.6, 14.4))  # 80% of (22, 18)
    ax = plt.gca()
    ax.set_aspect('auto')
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, autopct='%1.0f%%', colors=colors, startangle=90,
        textprops={'color':'#fff', 'fontsize':24}, explode=explode, shadow=True
    )
    for w in wedges:
        w.set_edgecolor('#232336')
    plt.title('Verified Purchase Ratio', fontsize=28, color='#22c55e')
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    # Set background to white for the pie chart
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=200, facecolor='white')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def get_product_reviews_and_ratings(product_name, return_rows=False):
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
    if return_rows:
        return reviews_and_ratings, product_rows
    return reviews_and_ratings

def analyze_product(product_name):
    # Get original reviews and ratings and product_rows
    reviews_and_ratings, product_rows = get_product_reviews_and_ratings(product_name, return_rows=True)
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

    # Pass user_reviews to all chart functions for real-time analytics
    plot_url = generate_bar_chart(ratings, sentiments, user_reviews=user_reviews)
    sentiment_time_url = generate_sentiment_over_time_chart(product_rows, user_reviews)
    verified_pie_url = generate_verified_pie_chart(product_rows, user_reviews)
    result = f"Average Rating: {avg_rating} | Average Sentiment: {avg_sentiment} | Recommendation: {recommendation}"
    return plot_url, result, avg_rating, avg_sentiment, recommendation, sentiment_time_url, verified_pie_url

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
    sentiment_time_url = None
    verified_pie_url = None

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
                plot_url, result, avg_rating, avg_sentiment, recommendation, sentiment_time_url, verified_pie_url = analyze_product(product_name)
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
                    sentiment_time_url=sentiment_time_url,
                    verified_pie_url=verified_pie_url,
                )
        product_name = request.form.get('product_name', '')
        if product_name:
            plot_url, result, avg_rating, avg_sentiment, recommendation, sentiment_time_url, verified_pie_url = analyze_product(product_name)
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
            sentiment_time_url=sentiment_time_url,
            verified_pie_url=verified_pie_url,
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
        sentiment_time_url=sentiment_time_url,
        verified_pie_url=verified_pie_url,
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
    sentiment_time_url = None
    verified_pie_url = None

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
        plot_url, result, avg_rating, avg_sentiment, recommendation, sentiment_time_url, verified_pie_url = analyze_product(product_name)
    return jsonify({
        'user_sentiment_score': user_sentiment_score,
        'user_sentiment_category': user_sentiment_category,
        'avg_rating': avg_rating,
        'avg_sentiment': avg_sentiment,
        'recommendation': recommendation,
        'plot_url': plot_url,
        'result': result,
        'sentiment_time_url': sentiment_time_url,
        'verified_pie_url': verified_pie_url,
    })

if __name__ == '__main__':
    nltk.download('punkt')
    app.run(port=8000, debug=True)
if __name__ == '__main__':
    nltk.download('punkt')
    app.run(port=8000, debug=True)
