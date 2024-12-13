from flask import Flask, render_template, request, jsonify
import joblib
import requests
import pandas as pd
import numpy as np

df = pd.read_csv('...\anime_final.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

theta = joblib.load('...\theta.pkl')

scaler = joblib.load('...\scaler.pkl')


def estimate_rank(row, non_nan_rows, k=4):
    non_nan_rows = non_nan_rows.copy()
    non_nan_rows['score_diff'] = abs(non_nan_rows['score'] - row['score'])
    closest_scores = non_nan_rows.nsmallest(k, 'score_diff')
    return round(closest_scores['rank'].mean())

def fetch_anime_info(anime_id):
    url = f'https://api.jikan.moe/v4/anime/{anime_id}/full'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Failed to fetch data for Anime ID: {anime_id}, Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data for Anime ID: {anime_id}: {e}")
        return None

def format_new_anime_data(anime_json):
    def format_data(entry):
        return ", ".join([item['name'] for item in entry]) if entry else "None"

    anime_formatted = {
        'title': anime_json['title'],
        'type': anime_json['type'],
        'source': anime_json['source'],
        'episodes': anime_json['episodes'],
        'rating': anime_json['rating'],
        'score': anime_json['score'],
        'rank': anime_json['rank'],
        'popularity': anime_json['popularity'],
        'favorites': anime_json['favorites'],
        'season': anime_json['season'],
        'year': anime_json['year'],
        'producers': format_data(anime_json.get('producers')),
        'licensors': format_data(anime_json.get('licensors')),
        'studios': format_data(anime_json.get('studios')),
        'genres': format_data(anime_json.get('genres')),
        'explicit_genres': format_data(anime_json.get('explicit_genres')),
        'themes': format_data(anime_json.get('themes')),
        'demographics': format_data(anime_json.get('demographics'))
    }

    return anime_formatted

def preprocess_anime_data(df, formatted_anime_df, split_and_encode, preprocess_custom, qt):

    original_features = df

    dummy_df = pd.DataFrame(0, index=np.arange(1), columns=original_features.columns)

    columns_to_encode = ['producers', 'genres', 'explicit_genres', 'themes']
    encoded_columns = split_and_encode(formatted_anime_df, columns_to_encode)

    encoded_columns = encoded_columns.rename(columns={"explicit_genres_None": "explicit_genres_nan"})

    formatted_anime_df.drop(columns_to_encode, axis=1, inplace=True)
    formatted_anime_df = pd.concat([formatted_anime_df, encoded_columns], axis=1)

    categorical_features = ['year', 'type', 'source', 'rating', 'season', 'licensors', 'studios', 'demographics']

    formatted_anime_df = pd.get_dummies(formatted_anime_df, columns=categorical_features, dtype=int)

    year_columns = [col for col in formatted_anime_df.columns if col.startswith('year_')]
    formatted_anime_df.rename(columns={col: f"{col}.0" for col in year_columns}, inplace=True)

    common_columns = dummy_df.columns.intersection(formatted_anime_df.columns)
    filtered_anime_df = formatted_anime_df[common_columns]

    df_features = pd.concat([dummy_df, filtered_anime_df], ignore_index=True)

    df_features = df_features.drop(0).reset_index(drop=True)

    nan_rows = df_features[df_features['rank'].isna()]
    non_nan_rows = df.dropna(subset=['rank'])
    for idx, row in nan_rows.iterrows():
        estimated_rank = estimate_rank(row, non_nan_rows)
        df_features.loc[idx, 'rank'] = estimated_rank

    if 'score_diff' in df_features.columns:
        df_features.drop(columns=['score_diff'], inplace=True)

    df_features = df_features.fillna(0)

    columns_to_standardize = ['episodes', 'score', 'rank', 'popularity', 'favorites']

    custom, _ = preprocess_custom(df_features, scaler, 'my_score', columns_to_standardize)

    return df_features, custom, formatted_anime_df['title']


def split_and_encode(df, column_names):
    encoded_dfs = []
    for column_name in column_names:
        df[column_name] = df[column_name].astype(str)
        encoded_df = df[column_name].str.get_dummies(sep=', ')
        encoded_df.columns = [f"{column_name}_{col}" for col in encoded_df.columns]
        encoded_dfs.append(encoded_df)
    return pd.concat(encoded_dfs, axis=1)

def preprocess_custom(df, scaler, target_column, columns_to_standardize, fit=False):
    df_processed = df.drop(['series_animedb_id', 'title'], axis=1)

    X = df_processed.drop(target_column, axis=1)
    y = 0

    if fit:
        X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])
    else:
        X[columns_to_standardize] = scaler.transform(X[columns_to_standardize])

    X_bias = np.c_[np.ones(X.shape[0]), X]

    return X_bias, y

def process_single_show(anime_id, df, split_and_encode, preprocess_custom, scaler, theta):
    raw_anime_data = fetch_anime_info(anime_id)

    if raw_anime_data:
        formatted_anime_data = format_new_anime_data(raw_anime_data)
        formatted_anime_df = pd.DataFrame([formatted_anime_data])
        custom_preprocessed_df, custom, title = preprocess_anime_data(df, formatted_anime_df, split_and_encode, preprocess_custom, scaler)
        # print(custom)
        y_pred_custom = custom.dot(theta)

        
        return y_pred_custom,title
    else:
        print("No data found for the given Anime ID.")
        return 0.0,"Not found"
    
# anime_id = 413
# single_show,title = process_single_show(anime_id, df, split_and_encode, preprocess_custom, scaler, theta)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  
        anime_id = int(data['anime_id'])  
        single_show,title = process_single_show(anime_id, df, split_and_encode, preprocess_custom, scaler, theta)
        return jsonify({'success': True, 'prediction': single_show.item(0),'title':title.item()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
