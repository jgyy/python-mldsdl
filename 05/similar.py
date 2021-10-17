"""
Finding Similar Movies
"""
from warnings import filterwarnings
from pandas import DataFrame, read_csv, merge
from numpy import size, mean


def wrapper():
    """
    wrapper function
    """
    filterwarnings("ignore", category=RuntimeWarning)
    filterwarnings("ignore", category=FutureWarning)
    r_cols = ["user_id", "movie_id", "rating"]
    ratings = DataFrame(
        read_csv(
            "ml-100k/u.data",
            sep="\t",
            names=r_cols,
            usecols=range(3),
            encoding="ISO-8859-1",
        )
    )
    m_cols = ["movie_id", "title"]
    movies = DataFrame(
        read_csv(
            "ml-100k/u.item",
            sep="|",
            names=m_cols,
            usecols=range(2),
            encoding="ISO-8859-1",
        )
    )
    ratings = merge(movies, ratings)
    print(ratings.head())
    movie_ratings = ratings.pivot_table(
        index=["user_id"], columns=["title"], values="rating"
    )
    print(movie_ratings.head())
    star_wars_ratings = movie_ratings["Star Wars (1977)"]
    print(star_wars_ratings.head())
    similar_movies = movie_ratings.corrwith(star_wars_ratings)
    similar_movies = similar_movies.dropna()
    dframe = DataFrame(similar_movies)
    print(dframe.head(10))
    print(similar_movies.sort_values(ascending=False))
    movie_stats = ratings.groupby("title").agg({"rating": [size, mean]})
    print(movie_stats.head())
    popular_movies = movie_stats["rating"]["size"] >= 100
    print(
        movie_stats[popular_movies].sort_values([("rating", "mean")], ascending=False)[
            :15
        ]
    )
    dframe = movie_stats[popular_movies].join(
        DataFrame(similar_movies, columns=["similarity"])
    )
    print(dframe.head())
    print(dframe.sort_values(["similarity"], ascending=False)[:15])


if __name__ == "__main__":
    wrapper()
