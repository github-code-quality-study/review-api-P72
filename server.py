import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
VALID_LOCATIONS = ["Albuquerque, New Mexico",
"Carlsbad, California",
"Chula Vista, California",
"Colorado Springs, Colorado"
"Denver, Colorado"
"El Cajon, California",
"El Paso, Texas",
"Escondido, California",
"Fresno, California",
"La Mesa, California",
"Las Vegas, Nevada",
"Los Angeles, California",
"Oceanside, California",
"Phoenix, Arizona",
"Sacramento, California",
"Salt Lake City, Utah",
"Salt Lake City, Utah",
"San Diego, California",
"Tucson, Arizona"
]
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Write your code here
            query_params = parse_qs(environ["QUERY_STRING"])
            location = query_params.get("location",[None])[0]
            start_date = query_params.get("start_date",[None])[0]
            end_date = query_params.get("end_date",[None])[0]

            filtered_reviews = reviews

            # Check if location query parameter is present and filter reviews by location
            if location:
                filtered_reviews = [review for review in filtered_reviews if review["Location"] == location]

            # Check if start_date and end_date query parameter is present and filter reviews by start_date & end_date
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date]
            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

            #  Apply sentiment analysis to each review
            filtered_reviews = list(map(lambda review: {**review, 'sentiment': self.analyze_sentiment(review['ReviewBody'])}, filtered_reviews))

            # sort reviews by sentiment by compound score descending order
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)  
            response_body = json.dumps(sorted_reviews,indent=2).encode("utf-8")
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size)
            try:
                data = parse_qs(request_body.decode('utf-8'))
                location = data.get('Location', [None])[0]
                review_body = data.get('ReviewBody', [None])[0]
            except Exception as e:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b'Invalid form data']


            # Check if location and review body is present in the request body
            if not location:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b'Missing location']
            if not review_body:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b'Missing review body']
            if location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Invalid location in request body"]
            
            # Analyze sentiment of the review body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            review_id = str(uuid.uuid4())

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp,
            }
            # Append the new review to the reviews list and update the reviews.csv file
            # reviews.append(new_review)
            # pd.DataFrame(reviews).to_csv('data/reviews.csv', index=False)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()