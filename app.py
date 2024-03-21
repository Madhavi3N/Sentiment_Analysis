from flask import Flask, render_template, request
import joblib
import re


app = Flask(__name__)

model = joblib.load('best_model/naive_bayes.pkl')

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r"\s+", ' ', text)
    return text

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        data_point = preprocess(review)

        prediction = model.predict([data_point])[0]

        sentiment = "😄 Positive Review 😄" if prediction == 'Positive' else "☹️ Negative Review ☹️"

        return render_template('pred.html', sentiment=sentiment, review=review)


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0", port=5000)