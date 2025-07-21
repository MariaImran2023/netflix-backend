import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Netflix Recommender Backend is running successfully!'

# Load models
with open('clustering_models.pkl', 'rb') as f:
    kmeans_movies, kmeans_shows = pickle.load(f)

# Load datasets
movies_df = pd.read_csv('movies.csv')
shows_df = pd.read_csv('shows.csv')

# Rating labels for description
rating_labels = {
    'TV-Y': 'TV-Y (All Children)',
    'TV-Y7': 'TV-Y7 (Ages 7+)',
    'TV-G': 'TV-G (All Ages)',
    'TV-PG': 'TV-PG (Ages 10+)',
    'TV-14': 'TV-14 (Ages 14+)',
    'TV-MA': 'TV-MA (Mature: Adults Only)',
    'G': 'G (All Ages)',
    'PG': 'PG (Ages 10+)',
    'PG-13': 'PG-13 (Ages 13+)',
    'R': 'R (Restricted: Ages 18+)',
    'NC-17': 'NC-17 (No Children Under 17)',
    'NR': 'NR (Not Rated)',
    'UR': 'UR (Unrated)'
}

# OMDB API Key
OMDB_API_KEY = '8833eb5f'

# Get IMDb Rating
def get_imdb_rating(title):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        response = requests.get(url).json()
        return response.get("imdbRating", "N/A")
    except:
        return "N/A"

# Group genres for dropdown
def get_grouped_genres():
    df = pd.concat([movies_df.assign(type='Movie'), shows_df.assign(type='TV Show')])
    df['listed_in'] = df['listed_in'].fillna("")

    genre_groups = {
        "Action": ["Action & Adventure", "TV Action & Adventure"],
        "Comedy": ["Comedies", "TV Comedies", "Stand-Up Comedy", "Stand-Up Comedy & Talk Shows"],
        "Drama": ["Dramas", "TV Dramas", "Teen TV Shows"],
        "Romance": ["Romantic Movies", "Romantic TV Shows"],
        "Horror": ["Horror Movies", "TV Horror"],
        "Sci-Fi": ["Sci-Fi & Fantasy", "TV Sci-Fi & Fantasy"],
        "Thriller": ["Thrillers", "TV Thrillers"],
        "Anime": ["Anime Features", "Anime Series"],
        "Documentary": ["Documentaries", "Docuseries"],
        "Family": ["Children & Family Movies", "Kids' TV"],
        "Reality": ["Reality TV"],
        "Music": ["Music & Musicals"],
        "Faith": ["Faith & Spirituality"],
        "Crime": ["Crime TV Shows", "Classic & Cult TV"],
        "International": ["International Movies", "International TV Shows", "Spanish-Language TV Shows", "Korean TV Shows"],
        "LGBTQ": ["LGBTQ Movies"],
        "Sports": ["Sports Movies"],
        "Classic": ["Classic Movies"]
    }

    def map_to_grouped_genres(original_genres):
        matched = set()
        for genre in original_genres:
            for group, members in genre_groups.items():
                if genre.strip() in members:
                    matched.add(group)
        return list(matched)

    df['genre_list'] = df['listed_in'].apply(lambda x: [i.strip() for i in x.split(',')])
    df['grouped_genres'] = df['genre_list'].apply(map_to_grouped_genres)

    movie_genres = sorted(set(g for sublist in df[df['type'] == 'Movie']['grouped_genres'] for g in sublist))
    tv_genres = sorted(set(g for sublist in df[df['type'] == 'TV Show']['grouped_genres'] for g in sublist))

    return movie_genres, tv_genres

# Recommend content
def recommend_content(content_type, genre, max_duration=None, min_rating=None):
    if content_type.lower() == 'movie':
        df = movies_df.copy()
        duration_col = 'duration_clean'
    elif content_type.lower() == 'tv show':
        df = shows_df.copy()
        duration_col = 'seasons'
    else:
        return []

    df['listed_in'] = df['listed_in'].fillna("")
    df = df[df['listed_in'].str.contains(genre, case=False, na=False)]

    if content_type.lower() == 'tv show':
        df = df[~df['title'].str.contains("Special|OVA|Movie|Episode", case=False, na=False)]

    if max_duration is not None:
        df = df[df[duration_col] <= max_duration]

    # Add rating description
    df['rating_desc'] = df['rating'].map(rating_labels).fillna('Unknown')

    # Get IMDb ratings
    df['imdb_rating'] = df['title'].apply(get_imdb_rating)
    df['imdb_rating_clean'] = pd.to_numeric(df['imdb_rating'], errors='coerce')

    # Filter by IMDb rating
    if min_rating is not None:
        min_rating = float(min_rating)
        df = df[df['imdb_rating_clean'].notna()]  
        df = df[df['imdb_rating_clean'] >= min_rating]

    # Sort by IMDb rating descending
    df = df.sort_values('imdb_rating_clean', ascending=False)

    return df[['title', 'rating_desc', 'duration', 'cluster', 'imdb_rating']]

# API for recommendation
@app.route('/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    content_type = data.get('type')
    genre = data.get('genre')
    max_duration = data.get('maxDuration')
    min_rating = data.get('minRating')

    results = recommend_content(content_type, genre, max_duration, min_rating)
    return jsonify(results.to_dict(orient='records'))

# API for genres
@app.route('/genres', methods=['GET'])
def api_genres():
    movie_genres, tv_show_genres = get_grouped_genres()
    return jsonify({
        "movies": movie_genres,
        "tv_shows": tv_show_genres
    })

if __name__ == '__main__':
    from os import environ
    port = int(environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
