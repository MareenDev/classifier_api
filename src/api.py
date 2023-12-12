from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def main():
    return {
        "hello": "world!!!123!!",
    }


@app.route('/hello_world', methods=['GET'])
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/trainings_data', methods=['GET'])
def get_trainingsdata():
    return Response(content_type='application/json')
