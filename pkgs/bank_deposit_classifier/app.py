from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        career = request.form['career']
        return f'Prediction for {career} is NOT_IMPLEMENTED'
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
