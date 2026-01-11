import numpy as np

import typing

#if typing.TYPE_CHECKING:
import sys
sys.path.append('..')

import neural_network as nn
import verification.nn_verification as nn_verif
import geometry.interval as interv



def get_random_vertex(
        lb  : np.ndarray,
        ub  : np.ndarray
    ) -> np.ndarray:
    """
        #### Description:
        Returns a *random* vertex of the `[lb, ub]` parallelepiped. Let `vertex` be
        a vertex. Then, the i-th coordinate is either `vertex_i = lb_i`, or
        `vertex_i = ub_i`. To make the latter choice each time we toss a coin.
    """

    assert lb.shape == ub.shape

    row_dim     = lb.shape[0]
    column_dim  = lb.shape[1]
    dim         = row_dim * column_dim

    flat_lb = lb.flatten()
    flat_ub = ub.flatten()

    # create a random binary vector
    r = np.random.randint(np.zeros(dim), 2 * np.ones(dim))
    R = np.diag(r)
    I = np.eye(dim)

    flat_vertex = np.matmul(R, flat_lb) + np.matmul((I - R), flat_ub)
    vertex      = flat_vertex.reshape([row_dim, column_dim])

    return vertex




# If I have done my job correctly, both the
# adversarial_attack() and the fidelity()
# methods work for both nn_verification and
# neural_network instances, i.e. nn_model can be both
def adversarial_attack(
        lb          : np.ndarray,
        ub          : np.ndarray,
        nn_model    : typing.Union[nn.MNISTNeuralNetwork, nn_verif.NNVerification],
        c_star      : int,
        num_samples : int = 1000,
        p           : float = 0.0
    ) -> typing.Union[np.ndarray, None]:
    """
        #### Description:
        It returns a *counterexample* using *sampling*. Namely, it returns a point
        in `[lb, ub] \ D_{c*}`. We compute `num_samples` samples. Each sample is
        either uniformly drwan from `[lb, ub]`, either consists from a *vertex* of
        the parallelepiped `[lb, ub]`. The later choice is made by tossing a coin.
        the parameter `p` denotes the probability that a samples comes from the
        parallelepiped's vertices.
    """

    assert lb.shape == ub.shape

    ## Try the bounds first!!
    #if c_star != nn_model.predict_argmax(lb)[0]: return lb
    #if c_star != nn_model.predict_argmax(ub)[0]: return ub

    for i in range(num_samples):
        # Toss a coin to get either a random vertex or a uniformly
        # distributed interior point
        if np.random.binomial(1, p) == 1:   X = get_random_vertex(lb, ub)
        else:                               X = np.random.uniform(lb, ub)
        y, _ = nn_model.predict_argmax(X)

        if c_star != y: return X
    
    return None



def fidelity(
        lb          : np.ndarray,
        ub          : np.ndarray,
        nn_model    : typing.Union[nn.MNISTNeuralNetwork, nn_verif.NNVerification],
        c_star      :int,
        num_samples :int =1000
    ) -> float:

    """
        #### Description:
        Let `[lb, ub]` be a *unsound* approximation to the decision surface
        of `c_star`. This function estimates the percision of `[lb, ub]` as
        a local approximation to the `nn_model` neural network. Namely, the
        number of the correctly classified examples, over the number of examples
        in `[lb, ub]`.

        fidelity = |[lb, ub] \ D_{c^\star}| / |[lb, ub]|

        #### Computation:
        1. Computes `num_samples` uniformly drawn random samples inside the
        `[lb, ub]` interval.
        2. Computes the number (`hits`) of the points in `[lb, ub]` that are 
        also classified as `c_star` by the neural network `nn_model`.
        3. Returns the ratio `hits`/`num_samples`.

        #### Notes:
        1. As `nn_model` can be passed any object that has the `predict_argmax()`
        defined.
        2. Naturally, it should hold `lb.shape == ub.shape`.
        3. For a *proper* stability guarantee, i.e. one gurantee that is
        returned by one of the algorithms in `search_algorithms`, this
        function is irrelevant. These intervals are *always* sound. The fidelity
        will *always* be 1.

        #### See also:
        * https://en.wikipedia.org/wiki/Precision_and_recall
    """

    assert lb.shape == ub.shape

    hits = 0
    for i in range(num_samples):
        X = np.random.uniform(lb, ub)
        y, _ = nn_model.predict_argmax(X)
        if c_star == y: hits += 1
    
    return hits/num_samples


def volume_approximation(
        interval    :interv.Interval,
        domain      :interv.Interval,
        num_samples :int =1_000_000
    ) -> float:

    """
        #### Description:
        A simple algorithm for volume approximation.

        vol = |`interval`| / |`domain`|

        #### Computation:
        1. We compute `num_samples` uniformly drawn random samples inside the
        `domain`.
        2. We count the number (`hits`) of samples that also belong to the `interval`.
        3. Returns: `hits`/`num_samples`.

        #### Notes:
        1. *Not* usefull in high dimensions, i.e. in the MNIST dataset. Result is
        *always* 0, due to *underflow*.

        #### See also:
        * https://en.wikipedia.org/wiki/Monte_Carlo_method
    """

    hits = 0
    for i in range(num_samples):
        X = np.random.uniform(domain.lb, domain.ub)
        if X in interval: hits += 1
    
    return hits/num_samples


def noise_analysis(
        lb          : np.ndarray,
        ub          : np.ndarray,
        nn_model    : typing.Union[nn.MNISTNeuralNetwork, nn_verif.NNVerification],
        num_classes : int,
        num_samples : int = 1000
    ):
    """
        #### Description:
        Estimating the distribution of uniformly drawn random samples from the
        `[lb, ub]` domain, to the `num_classes` classes by the neural network
        `nn_model`.
    """

    histogram = np.zeros(num_classes)

    samples = []
    for i in range(num_samples):
        X = np.random.uniform(lb, ub)
        samples.append(X)
        prediction = nn_model.predict_argmax(X)[0]
        histogram[prediction] += 1
    
    return histogram, samples



def evaluate(
        nn_model    : typing.Union[nn.MNISTNeuralNetwork, nn_verif.NNVerification],
        X           : np.ndarray,
        Y           : np.ndarray,
        shuffle     : bool  = False,
        num_samples : int   = 1000
    ):
    """
        #### Description:
        Approximate the `nn_model` neural networks accuracy, on `num_samples` of the
        dataset `X` with the labels in `Y`. If `shuffle` is True, we shuffle `X`
        before choosing the sumples.
    """

    assert num_samples > 0
    assert len(Y) > num_samples

    ## If needed, shuffle
    sample_indices = list(range(num_samples))
    if shuffle: np.random.shuffle(sample_indices)
    predicted_labels = []
    hits = 0
    for ind in sample_indices:
        prediction, _ = nn_model.predict_argmax(X[ind, :])
        predicted_labels.append(prediction)
        if prediction == Y[ind]: hits += 1
    
    accuracy = hits/num_samples
    
    return accuracy, predicted_labels


## Constants
pixels_uniform  = 0
pixel_normal    = 1
pixels_dom_min  = 2
pixels_dom_max  = 3

def pixel_noise(
        max_num_pixels  : int,
        row_dim         : int,
        col_dim         : int,
        dom_min         : float,
        dom_max         : float,
        op              : int= 0
    ):
    """
        #### Description:
        We construct a `row_dim x col_dim` matrix, with `max_num_pixels` non-zero
        entries. These entries are randomly uniformly drawn. The value of each
        non-zero pixel is drawn from the `[dom_min, dom_max]` interval. The `op`
        parameter controls the way we choose the latter value.

        * `op == 0`: Uniformly drawn.
        * `op == 1`: The middle of the domain.
        * `op == 2`: The min value, `dom_min`.
        * `op == 3`: The max value `dom_max`.
    """
    noise = np.zeros([row_dim, col_dim])

    for num_pixels in range(max_num_pixels):
        i = np.random.randint(0, row_dim-1)
        j = np.random.randint(0, col_dim-1)

        if op == pixels_uniform:
            n = np.random.uniform(dom_min, dom_max)
        elif op == pixel_normal:
            n = np.random.normal(dom_min + (dom_max - dom_min)/2, 0.1)
        elif op == pixels_dom_min:
            n = dom_min
        elif op == pixels_dom_max:
            n = dom_max
        else:
            raise Exception("montecarlo.utilities.pixel_noise: Unkown op code")

        if noise[i][j] == 0: noise[i][j] += n
    

    return noise