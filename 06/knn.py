"""
KNN (K-Nearest-Neighbors)
"""
from operator import itemgetter
from pandas import DataFrame, read_csv
from numpy import size, mean, min as npmin, max as npmax, array
from scipy.spatial import distance


def compute_distance(amovie, bmovie):
    """
    Function that computes the "distance" between two movies based on how
    similar their genres are, and how similar their popularity is.
    """
    genres_a = amovie[1]
    genres_b = bmovie[1]
    genre_distance = distance.cosine(genres_a, genres_b)
    popularity_a = amovie[2]
    popularity_a = bmovie[2]
    popularity_distance = abs(popularity_a - popularity_a)
    return genre_distance + popularity_distance


def get_neighbors(movie_id, knn, movie_dict):
    """
    Sort those by distance, and print out the K nearest neighbors
    """
    distances = []
    for movie in movie_dict:
        if movie != movie_id:
            dist = compute_distance(movie_dict[movie_id], movie_dict[movie])
            distances.append((movie, dist))
    distances.sort(key=itemgetter(1))
    neighbors = []
    for xval in range(knn):
        neighbors.append(distances[xval][0])
    return neighbors


def wrapper():
    """
    wrapper function
    """
    r_cols = ["user_id", "movie_id", "rating"]
    ratings = DataFrame(
        read_csv("ml-100k/u.data", sep="\t", names=r_cols, usecols=range(3))
    )
    print(ratings.head())
    movie_properties = ratings.groupby("movie_id").agg({"rating": [size, mean]})
    print(movie_properties.head())
    movie_num_ratings = DataFrame(movie_properties["rating"]["size"])
    movie_normalized_num_ratings = movie_num_ratings.apply(
        lambda x: (x - npmin(x)) / (npmax(x) - npmin(x))
    )
    print(movie_normalized_num_ratings.head())
    movie_dict = {}
    with open(r"ml-100k/u.item", encoding="ISO-8859-1") as file:
        for line in file:
            fields = line.rstrip("\n").split("|")
            movie_id = int(fields[0])
            name = fields[1]
            genres = fields[5:25]
            genres = map(int, genres)
            movie_dict[movie_id] = (
                name,
                array(list(genres)),
                movie_normalized_num_ratings.loc[movie_id].get("size"),
                movie_properties.loc[movie_id].rating.get("mean"),
            )
    print(movie_dict[1])
    print(compute_distance(movie_dict[2], movie_dict[4]))
    print(movie_dict[2])
    print(movie_dict[4])
    avg_rating = 0
    neighbors = get_neighbors(1, 10, movie_dict)
    for neighbor in neighbors:
        avg_rating += movie_dict[neighbor][3]
        print(movie_dict[neighbor][0] + " " + str(movie_dict[neighbor][3]))
    avg_rating /= 10
    print(avg_rating)
    print(movie_dict[1])


if __name__ == "__main__":
    wrapper()
