from flask import Flask, render_template, request
from bank_deposit_classifier.predict import (load_model, predict_with_defaults,
                                            CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)


app = Flask(__name__)
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        categoicals = [x for x in request.form.values() if x in CATEGORICAL_COLUMNS]
        numerics = {x:request.form.get(x) for x in request.form if x in NUMERIC_COLUMNS}
        prediction = predict_with_defaults(
            model,
            categoricals=categoicals,
            numerics=numerics
            )
        prediction_ = 'WILL' if prediction == 1 else 'WILL NOT'
        return f'We predict the customer {prediction_} submit a deposit'

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)
