# 🛒 BuyWise Product Analysis

A modern, full-stack web application for analyzing and reviewing products using real customer feedback, advanced sentiment analysis, and interactive data visualizations. BuyWise is designed to help users make smarter shopping decisions by leveraging AI and NLP to extract actionable insights from multilingual reviews.

---

## 🚀 Features

- 📊 **Product Analytics Dashboard**: Instantly visualize product ratings, sentiment trends, and verified purchase ratios with interactive charts.
- 🤖 **AI-Powered Sentiment Analysis**: Automatically detects the sentiment (positive, negative, neutral) of reviews in any language using Google Translate and TextBlob.
- 🌍 **Multilingual Support**: Reviews in any language are translated and analyzed for global coverage.
- 📝 **Customer Reviews**: Users can submit their own reviews, see their sentiment score, and watch their feedback impact the analytics in real time.
- 🏆 **Top Review Highlight**: The highest-rated review (including user submissions) is always featured at the top.
- 🔒 **Verified Reviews Only**: Only verified purchase reviews are included in analytics for accuracy and trust.
- ⚡ **Live Updates**: All analytics, charts, and recommendations update instantly when a new review is submitted.
- 💡 **Smart Recommendation**: The app provides a clear "Buy" or "Don't Buy" recommendation based on aggregated sentiment and ratings.
- 🎨 **Modern, Responsive UI**: Built with Bootstrap 5, Animate.css, and custom CSS for a visually appealing and mobile-friendly experience.

---

## 🌟 Project Highlights

- **End-to-End Solution:** Combines data science, NLP, and web development in a single, production-ready app.
- **Real-Time Analytics:** User reviews are processed and reflected in all charts and recommendations without page reloads.
- **Asynchronous Translation:** Uses Google Translate API for seamless multilingual sentiment analysis.
- **Data Visualization:** Matplotlib generates high-quality, dynamic charts for ratings and sentiment trends.
- **Recruiter-Friendly:** Clean, modular codebase with clear separation of backend (Flask/Python) and frontend (HTML/CSS/JS).

---

## 🛠️ Technologies Used

- **Backend:** Python, Flask, Pandas, Matplotlib, TextBlob, NLTK, Googletrans
- **Frontend:** HTML5, CSS3, Bootstrap 5, Animate.css, JavaScript (ES6+)
- **AJAX:** Real-time review submission and UI updates via Fetch API
- **Data:** CSV-based dataset of Amazon product reviews (see `dataset.csv`)
- **Deployment:** Runs locally with a single Python script (`app.py`)

---

## 📦 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Buywise-Product-Analysis.git
   cd Buywise-Product-Analysis
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000).

---

## 📝 Usage

- **Search for a product:** Start typing in the product name field for instant suggestions.
- **View analytics:** See rating distributions, sentiment trends, and recommendations for the selected product.
- **Submit a review:** Add your own review and see its sentiment score and impact on all analytics instantly.
- **Top review:** The highest-rated review (including yours) is always highlighted.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📁 Project Structure

- `app.py` — Flask backend, sentiment analysis, chart generation, and AJAX endpoints.
- `templates/index.html` — Main UI, Bootstrap/Animate.css, AJAX logic for live updates.
- `static/js/legacy-customer-review.js` — (Legacy) JS for client-side review storage (not used in current app).
- `dataset.csv` — Amazon product reviews dataset.
- `static/css/styles.css` — Custom styles (optional, referenced in HTML).
- `README.md` — This file.

---

## 🎯 Conclusion

BuyWise demonstrates the integration of data analysis, NLP, and modern web development to create a robust, interactive product review analysis tool. It is an excellent showcase for recruiters interested in full-stack engineering, data science, and real-world NLP applications.

Explore the code, try the live analytics, and see how AI can power smarter shopping!

---


