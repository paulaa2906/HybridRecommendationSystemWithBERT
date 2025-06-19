import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as fancybboxpatch
import seaborn as sns
import datetime
from wordcloud import WordCloud
import re
import warnings
from typing import Tuple, Dict
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
warnings.filterwarnings('ignore')

# Setting style for visualization
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')

def create_text_profile(df_movies, df_ratings, df_sentiment):
    """Creates text-based profile report"""
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    # Movies Dataset
    print("\nMOVIES DATASET")
    print("-" * 20)
    print(f"Rows: {df_movies.shape[0]}")
    print(f"Columns: {df_movies.shape[1]}")
    print("\nColumn Types:")
    print(df_movies.dtypes)
    print("\nMissing Values:")
    print(df_movies.isnull().sum())
    
    # Add statistical analysis for movies
    movies_stats = calculate_statistics(df_movies)
    if movies_stats:
        print("\nStatistical Analysis:")
        print("-" * 20)
        for col, metrics in movies_stats.items():
            print(f"\nColumn: {col}")
            for metric, value in metrics.items():
                print(f"{metric:>8}: {value:>.2f}")
    
    # Ratings Dataset
    print("\nRATINGS DATASET")
    print("-" * 20)
    print(f"Rows: {df_ratings.shape[0]}")
    print(f"Columns: {df_ratings.shape[1]}")
    print("\nColumn Types:")
    print(df_ratings.dtypes)
    print("\nMissing Values:")
    print(df_ratings.isnull().sum())
    
    # Add statistical analysis for ratings
    ratings_stats = calculate_statistics(df_ratings)
    if ratings_stats:
        print("\nStatistical Analysis:")
        print("-" * 20)
        for col, metrics in ratings_stats.items():
            print(f"\nColumn: {col}")
            for metric, value in metrics.items():
                print(f"{metric:>8}: {value:>.2f}")
    
    # Sentiment Dataset
    print("\nSENTIMENT DATASET")
    print("-" * 20)
    print(f"Rows: {df_sentiment.shape[0]}")
    print(f"Columns: {df_sentiment.shape[1]}")
    print("\nColumn Types:")
    print(df_sentiment.dtypes)
    print("\nMissing Values:")
    print(df_sentiment.isnull().sum())
    
    # Add statistical analysis for sentiment
    sentiment_stats = calculate_statistics(df_sentiment)
    if sentiment_stats:
        print("\nStatistical Analysis:")
        print("-" * 20)
        for col, metrics in sentiment_stats.items():
            print(f"\nColumn: {col}")
            for metric, value in metrics.items():
                print(f"{metric:>8}: {value:>.2f}")

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate basic statistics for numeric columns"""
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew()
        }
    return stats

def create_correlation_heatmap(df: pd.DataFrame, title: str) -> None:
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap: {title}')
    plt.tight_layout()
    plt.show()

def analyze_temporal_patterns(df_ratings: pd.DataFrame) -> None:
    """Analyze temporal patterns in ratings"""
    if 'timestamp' in df_ratings.columns:
        df_ratings['datetime'] = pd.to_datetime(df_ratings['timestamp'], unit='s')
        
        # Add time-based features
        df_ratings['hour'] = df_ratings['datetime'].dt.hour
        df_ratings['day_of_week'] = df_ratings['datetime'].dt.day_name()
        df_ratings['month'] = df_ratings['datetime'].dt.month_name()
        
        # Create a figure with two subplots stacked vertically and increased spacing
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'hspace': 0.6})
        
        # Define custom colors
        low_color = '#F2E2B1'
        mid_color = 'brown'
        high_color = '#1F7D53'
        
        # For hourly distribution
        counts = df_ratings['hour'].value_counts().sort_index()
        
        # Find thresholds for color assignment
        min_count = counts.min()
        max_count = counts.max()
        range_count = max_count - min_count
        low_threshold = min_count + range_count / 3
        high_threshold = min_count + 2 * range_count / 3
        
        # Assign colors based on actual count values
        palette = {}
        for hour, count in counts.items():
            if count < low_threshold:
                palette[hour] = low_color
            elif count < high_threshold:
                palette[hour] = mid_color
            else:
                palette[hour] = high_color
        
        # Plot with the first axis
        bars = sns.countplot(data=df_ratings, x='hour', palette=palette, ax=ax1)
        ax1.set_title('Distribution of Ratings by Hour', fontsize=12, pad=30)
        ax1.set_xticklabels(ax1.get_xticklabels())
        
        # Add value labels
        for p in bars.patches:
            ax1.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', c='grey', fontsize='x-small')
        
        # For daily distribution
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        counts = df_ratings['day_of_week'].value_counts()
        counts = counts.reindex(day_order)
        
        # Assign colors based on actual count values
        palette = {}
        for day, count in counts.items():
            if count < low_threshold:
                palette[day] = low_color
            elif count < high_threshold:
                palette[day] = mid_color
            else:
                palette[day] = high_color
        
        # Plot with the second axis
        bars = sns.countplot(data=df_ratings, x='day_of_week', order=day_order, 
                          palette=palette, ax=ax2)
        ax2.set_title('Distribution of Ratings by Day of Week', fontsize=14, pad=30)
        
        # Make day name fonts smaller and ensure they don't get cropped
        ax2.set_xticklabels(ax2.get_xticklabels(),fontsize=12, ha='center')

        # Add value labels
        for p in bars.patches:
            ax2.annotate(f'{int(p.get_height())}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', c='grey', fontsize='x-small')
        
        # Add more bottom padding to avoid day_of_week text getting cropped
        plt.subplots_adjust(bottom=0.15)
        
        plt.tight_layout()
        plt.show()

def create_visual_profile(df_movies, df_ratings, df_sentiment):
    """Creates visualization-based profile report"""
    # Create multiple figures instead of one large figure
    
    # 1. Movies Analysis
    plt.figure(figsize=(15, 10))
    plt.suptitle('Movies Dataset Analysis', fontsize=16)
    
    # Extract years and create year plot
    df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)').astype('str')
    year_counts = df_movies['year'].value_counts().sort_index()
    
    plt.subplot(2, 1, 1)
    
    # Define color thresholds for years
    min_count = year_counts.min()
    max_count = year_counts.max()
    range_count = max_count - min_count
    low_threshold = min_count + range_count / 3
    high_threshold = min_count + 2 * range_count / 3
    
    # Define colors
    low_color = '#F2E2B1'
    mid_color = 'brown'
    high_color = '#1F7D53'
    
    # Create year palette
    year_palette = {}
    for year, count in year_counts.items():
        if count < low_threshold:
            year_palette[year] = low_color
        elif count < high_threshold:
            year_palette[year] = mid_color
        else:
            year_palette[year] = high_color
    
    # Create bar plot for years with color palette
    ax = sns.barplot(x=year_counts.index.astype(str), 
                    y=year_counts.values,
                    palette=year_palette)
    
    plt.title('Number of Movies by Year')
    plt.xticks(rotation=90, fontsize=8, fontfamily='monospace')
    plt.xlabel('Year')
    plt.ylabel('Count')
    
    # Add value labels on bars with improved styling
    for i, v in enumerate(year_counts.values):
        ax.text(i, v, str(v), 
                ha='center', 
                va='bottom',
                size='6',
                family='serif',
                style='italic',
                color='grey')
    
    # Genre analysis with ordered display
    genres_list = []
    for g in df_movies['genres']:
        genres_list.extend(g.split('|'))
    genres_counts = pd.Series(genres_list).value_counts()
    
    plt.subplot(2, 1, 2)
    # Define order for top genres (most common to least common)
    genre_order = genres_counts.head(15).index.tolist()
    top_genres = genres_counts.reindex(genre_order)

    # Define color thresholds
    min_count = top_genres.min()
    max_count = top_genres.max()
    range_count = max_count - min_count
    low_threshold = min_count + range_count / 3
    high_threshold = min_count + 2 * range_count / 3
    
    palette = {}
    for genre, count in top_genres.items():
        if count < low_threshold:
            palette[genre] = low_color
        elif count < high_threshold:
            palette[genre] = mid_color
        else:
            palette[genre] = high_color    

    # Create bar plot for genres with ordered display
    ax = sns.barplot(x=top_genres.values, 
                    y=top_genres.index,
                    palette=palette,
                    order=genre_order)
    plt.title('Top 15 Movie Genres')
    plt.xlabel('Count')
    
    # Add value labels on genre bars
    for i, v in enumerate(top_genres):
        ax.text(v, i, f' {v}', va='center', c= 'grey', fontsize='small')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Ratings Analysis
    plt.figure(figsize=(15, 10))
    plt.suptitle('Ratings Dataset Analysis', fontsize=16)
    
    plt.subplot(2, 1, 1)
    # Create histogram for ratings distribution
    ax = sns.histplot(df_ratings['rating'], bins=10, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Add value labels on rating distribution bars
    for i in ax.patches:
        ax.text(
            i.get_x() + i.get_width()/2,
            i.get_height(),
            int(i.get_height()),
            ha='center',
            va='bottom'
        )
    
    plt.subplot(2, 1, 2)
    user_rating_counts = df_ratings['userId'].value_counts()
    sns.histplot(user_rating_counts, bins=30, kde=True)
    plt.title('Distribution of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    
    # 3. Sentiment Analysis
    plt.figure(figsize=(15, 10))
    plt.suptitle('Sentiment Analysis', fontsize=16)
    
    plt.subplot(2, 1, 1)
    sns.histplot(df_sentiment['positive_probability'], bins=20, kde=True, label='Positive')
    sns.histplot(df_sentiment['negative_probability'], bins=20, kde=True, label='Negative', 
                color='red', alpha=0.6)
    plt.title('Distribution of Sentiment Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    sentiment_percentages = pd.DataFrame({
        'Percentage': [df_sentiment['positive_percent'].mean(), 
                      df_sentiment['negative_percent'].mean()],
        'Type': ['Positive', 'Negative']
    })
    sns.barplot(data=sentiment_percentages, x='Type', y='Percentage')
    plt.title('Average Sentiment Distribution')
    plt.ylabel('Percentage (%)')
    plt.tight_layout()
    plt.show()

def create_wordcloud(df_movies: pd.DataFrame) -> None:
    """Generate and display a word cloud from movie titles or descriptions"""
    # Combine all text into a single string
    text = ' '.join(df_movies['title'].astype(str))  # for movie titles
    # or
    # text = ' '.join(df_movies['description'].astype(str))  # for movie descriptions

    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         max_words=200,
                         max_font_size=40).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def create_movie_dataset_profiling(df_movies: pd.DataFrame, 
                                 df_ratings: pd.DataFrame, 
                                 df_sentiment: pd.DataFrame) -> None:
    """Main function to create comprehensive profile report"""
    # # Create text profile first
    create_text_profile(df_movies, df_ratings, df_sentiment)
    
    print("\nGenerating Correlation Analysis...")
    # Create correlation heatmaps
    create_correlation_heatmap(df_ratings, "Ratings")
    create_correlation_heatmap(df_sentiment, "Sentiment")
    
    print("\nGenerating Temporal Analysis...")
    # Analyze temporal patterns
    analyze_temporal_patterns(df_ratings)
    
    print("\nGenerating Visualizations...")
    # # Create visual profile
    create_visual_profile(df_movies, df_ratings, df_sentiment)
    
    # print("\nGenerating Word Cloud...")
    # Create word cloud
    # create_wordcloud(df_movies)