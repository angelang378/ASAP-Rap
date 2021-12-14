from flask import Flask, render_template, request, redirect, url_for
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
        user1playlist = save_playlist(request.form['user1playlist'])
        user2playlist = save_playlist(request.form['user2playlist'])
        train_model(user1playlist, user2playlist, user1name, user2name)
        return redirect(url_for('view_recommend_page'))
    else:
        return render_template("analyze.html", title="Analyze Taste")

@app.route("/recommend", methods=['GET', 'POST'])
def view_recommend_page():
    if request.method == 'POST':
        data = save_playlist(request.form['recplaylist'])
        mpath = "trained_models/" + request.form['model'] + ".pt"
        output = predict(data, mpath)
        return view_results_page(output)
    else:
        return render_template("recommend.html", title="Recommend Music", models=get_models())


@app.route("/results")
def view_results_page(output):
    return render_template("results.html", title="Resulting Recommendations", output=output)
