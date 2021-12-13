from flask import Flask, render_template, request
from music_data import *
from model import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/analyze", methods=['GET', 'POST'])
def view_analyze_page():
    if request.method == 'POST':
        user1name=request.form['user1name']
        user2name=request.form['user2name']
        print("getting user1playlist ")
        user1playlist = save_playlist(request.form['user1playlist'])
        print("getting user2playlist ")
        user2playlist = save_playlist(request.form['user2playlist'])
        print("training model")
        train_model(user1playlist, user2playlist, user1name, user2name)
        print("trained")
        return render_template('recommend.html',title="Recommend Music",
            user1name=request.form['user1name'],
            user1playlist=request.form['user1playlist'],
            user2name=request.form['user2name'],
            user2playlist=request.form['user2playlist'])
    else:
        return render_template("analyze.html", title="Analyze Taste")

@app.route("/recommend", methods=['GET', 'POST'])
def view_recommend_page():
    return render_template("recommend.html", title="Recommend Music")