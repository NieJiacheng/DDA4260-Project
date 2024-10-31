import numpy as np
import projectLib as lib

# set highest rating
K = 5

def softmax(x) -> np.ndarray:
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmaxLastAxis(x) -> np.ndarray:
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units (j)
    # F is the number of hidden units (i)
    # K is the highest rating (fixed to 5 here) (k)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x):
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    return np.divide(
            np.exp(x),
            np.ones(x.shape) + np.exp(x)
            )

def visibleToHiddenVec(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    
    # suppose v is one-hot
    
    probability_h_eq_1_given_v = sig(
        np.einsum("jik, jk->i", w, v)
    )
    
    # naive implementation
    naive_einsum = np.zeros(w.shape[1])
    for i in range(w.shape[1]):
        for j in range(w.shape[0]):
            for k in range(K):
                naive_einsum[i] += w[j, i, k] * v[j, k]
    
    naive_probability_h_eq_1_given_v = sig(naive_einsum)

    assert np.isclose(naive_probability_h_eq_1_given_v, probability_h_eq_1_given_v).all()
    
    return probability_h_eq_1_given_v

def hiddenToVisible(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    
    probability_vjk_eq_1_given_h = softmaxLastAxis(
        np.einsum("jik, i -> jk", w, h)
    )
    
    assert list(probability_vjk_eq_1_given_h.shape) == list([int(w.shape[0]), K])
    
    naive_einsum = np.zeros((w.shape[0], K))
    for j in range(w.shape[0]):
        for k in range(K):
            for i in range(w.shape[1]):
                naive_einsum[j, k] += w[j, i, k] * h[i]
        
    naive_probability_vjk_eq_1_given_h = softmaxLastAxis(naive_einsum)
    
    assert np.isclose(naive_probability_vjk_eq_1_given_h, probability_vjk_eq_1_given_h).all()
    
    return probability_vjk_eq_1_given_h

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    
    # propagate user input to hidden units
    dist_h = visibleToHiddenVec(v, w)
    
    sampled_h = sample(dist_h)
    
    # backpropagate hidden states to obtain distribution over movie q
    dist_v = hiddenToVisible(sampled_h, np.expand_dims(wq, axis=0))
    
    dist_v = dist_v.squeeze(axis=0)
    
    assert list(dist_v.shape) == 1
    assert dist_v.shape[0] == K
    
    return dist_v

def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    
    # not only for one user, but can be use for all users in a batch
    
    
    return np.argmax(ratingDistribution, axis=-1) + 1 # + 1 to convert from 0-4 to 1-5


def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution
    
    # not only for one user, but can be use for all users in a batch
    
    ratings_of_same_shape = np.arange(1, K+ 1, 1)
    
    weighted_average_rating = np.sum(np.multiply(ratingDistribution, ratings_of_same_shape), axis=-1)
    
    return weighted_average_rating

def predictMovieForUser(q, user, W, allUsersRatings, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = allUsersRatings[user]
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, allUsersRatings, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, allUsersRatings, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, allUsersRatings, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    return [predictMovieForUser(movie, user, W, allUsersRatings, predictType=predictType) for movie in range(W.shape[0])]
