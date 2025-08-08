# Sentiment Stock Forecaster 📈

Welcome to the Sentiment Stock Forecaster! This is a full-stack web application designed to predict the weekly direction of stock prices by combining traditional time-series analysis with modern Natural Language Processing (NLP) for sentiment analysis of financial news. The application features a sleek, user-friendly interface where you can select a stock and instantly receive a prediction and a series of insightful visualizations.

---

## Features

* **AI-Powered Prediction**: Utilizes a sophisticated **XGBoost** machine learning model to forecast whether a stock will close higher or lower than the previous week.
* **Real-Time Sentiment Analysis**: Fetches the latest financial news for a given stock and uses **NLP (VADER)** to analyze the sentiment of the headlines, providing a score of the overall market mood.
* **Dynamic Visualizations**: Generates three clear and informative charts using **Matplotlib**:
    1.  A 30-day price history chart.
    2.  A 30-day sentiment history chart.
    3.  A merged, dual-axis chart comparing price action directly against daily sentiment.
* **Interactive Frontend**: A clean and modern user interface built with **vanilla JavaScript** allows users to select stocks from categorized dropdowns (by industry).
* **Efficient Caching**: Employs a **SQLite** database to cache the results of daily predictions, preventing redundant API calls and ensuring a fast user experience on subsequent requests for the same stock.

---

## Tech Stack

* **Backend**: Python, Flask
* **Machine Learning**: XGBoost, Pandas, Scikit-learn
* **Data & APIs**: yfinance (Yahoo Finance), NewsAPI
* **Frontend**: HTML, CSS, Vanilla JavaScript
* **Database**: SQLite

---

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Installation & Setup

1.  **Clone the repository** (or download the files into a project folder):
    ```bash
    git clone https://github.com/AnooshVenkat/Market_Pulse.git
    cd Market_Pulse
    ```

2.  **Create and activate a virtual environment**:
    * **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required packages**:
    ```bash
    pip install Flask pandas-ta xgboost newsapi-python nltk yfinance
    ```

4.  **Download the NLTK VADER lexicon**:
    Open a Python interpreter in your terminal by typing `python` and then run the following commands:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    exit()
    ```

5.  **Configure Your API Key**:
    * Sign up for a free API key at [**NewsAPI.org**](https://newsapi.org/).
    * Open the `main.py` file and find the following line:
        ```python
        NEWS_API_KEY = ''
        ```
    * Place the actual key you received.

### Running the Application

1.  From your project's root directory, run the Flask application:
    ```bash
    python main.py
    ```

2.  Open your web browser and navigate to the following address:
    [**http://127.0.0.1:5000**](http://127.0.0.1:5000)

You should now see the application running! Select an industry and a company, then click "Predict" to see the analysis.
