from flask import Flask, render_template, request
from bank_deposit_classifier.predict import load_model, predict_with_defaults


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        career = request.form['career']
        model = load_model()
        prediction = predict_with_defaults(
            model,
            categoricals=[career]
            )
        return f'Prediction for {career} is {prediction}!'
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
