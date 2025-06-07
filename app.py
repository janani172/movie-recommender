from flask import Flask, render_template, request, jsonify
from recommender import recommend, new_df

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    movie_title = None
    recommendations = []
    if request.method == 'POST':
        movie_title = request.form['movie']
        recommendations = recommend(movie_title)
    return render_template('index.html', movie_title=movie_title, recommendations=recommendations)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '').lower()
    suggestions = []
    if query:
        suggestions = [title for title in new_df['title'] if title.lower().startswith(query)]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)