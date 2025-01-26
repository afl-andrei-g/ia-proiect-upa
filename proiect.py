from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home_page():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Aplicatie web</title> 
    </head>
    <body>
        <h1>Incarca CSV pentru analiza</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Incarca</button>
        </form>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nu ai selectat un fisier'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nu ai selectat un fisier'})

        data = pd.read_csv(file)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return jsonify({
            'message': 'Datele au fost procesate cu succes.',
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
