"""
Item-Based Collaborative Filtering
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv, merge, Series


def wrapper():
    """
    wrapper function
    """
    r_cols = ["user_id", "movie_id", "rating"]
    ratings = DataFrame(
        read_csv(
            join(dirname(__file__), "ml-100k/u.data"),
            sep="\t",
            names=r_cols,
            usecols=range(3),
            encoding="ISO-8859-1",
        )
    )
    m_cols = ["movie_id", "title"]
    movies = DataFrame(
        read_csv(
            join(dirname(__file__), "ml-100k/u.item"),
            sep="|",
            names=m_cols,
            usecols=range(2),
            encoding="ISO-8859-1",
        )
    )
    ratings = merge(movies, ratings)
    print(ratings.head())
    user_ratings = ratings.pivot_table(
        index=["user_id"], columns=["title"], values="rating"
    )
    print(user_ratings.head())
    corr_matrix = user_ratings.corr()
    print(corr_matrix.head())
    corr_matrix = user_ratings.corr(method="pearson", min_periods=100)
    print(corr_matrix.head())
    my_ratings = user_ratings.loc[0].dropna()
    print(my_ratings)
    sim_candidates = Series(dtype="float64")
    for i, j in enumerate(my_ratings.index):
        print("Adding sims for " + j + "...")
        sims = corr_matrix[j].dropna()
        sims = sims.map(lambda x, y=my_ratings[i]: x * y)
        sim_candidates = sim_candidates.append(sims)
    print("sorting...")
    sim_candidates.sort_values(inplace=True, ascending=False)
    print(sim_candidates.head(10))
    sim_candidates = sim_candidates.groupby(sim_candidates.index).sum()
    sim_candidates.sort_values(inplace=True, ascending=False)
    print(sim_candidates.head(10))
    filtered_sims = sim_candidates.drop(my_ratings.index)
    print(filtered_sims.head(10))


if __name__ == "__main__":
    wrapper()
