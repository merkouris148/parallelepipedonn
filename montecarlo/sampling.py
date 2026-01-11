## 3rd party libraries
import numpy as np

## custom libraries




gaussian    = 0
uniform     = 1

distributions = [
    gaussian,
    uniform
]

class Sampler(dict):

    ## Initialization
    def __init__(
            self,
            x_star,
            domain,
            nn_model,
            num_samples     = 1_000_000_000,
            distribution    = gaussian
    ):
        assert num_samples > 0
        assert x_star in domain
        assert distribution in distributions
        
        self.x_star         = x_star
        self.row_dim        = x_star.shape[0]
        self.column_dim     = x_star.shape[0]
        self.domain         = domain
        self.nn_model       = nn_model
        self.num_samples    = num_samples
        self.distribution   = distribution


        if self.distribution == gaussian:   self.sample_gaussian()
        else:                               self.sample_uniform()


    def sample_uniform(self):
        assert self.distribution == uniform

        for i in range(self.num_samples):
            # pick a random point ~ normal distribution
            X = np.random.uniform(self.domain.lb, self.domain.ub)
            
            # get the class of the random point
            c = self.nn_model.predict_argmax(X)[0]

            # sort the random poitn to the dictionary
            if c not in self.keys():    self[c] = [X]
            else:                       self[c].append(X)
    

    def sample_gaussian(self):
        assert self.distribution == gaussian

        # compute a reasonable standard deviation
        sigma = self.domain.get_diameter() / 100
        print(sigma)

        for i in range(self.num_samples):
            # pick a random point ~ normal distribution
            X = np.random.normal(self.x_star, (sigma**2) * np.ones([self.row_dim, self.column_dim]))
            X = np.abs(X)
            # we skip points outside of the domain
            if X not in self.domain: continue
            
            # get the class of the random point
            c = self.nn_model.predict_argmax(X)[0]

            # sort the random poitn to the dictionary
            if c not in self.keys():    self[c] = [X]
            else:                       self[c].append(X)

            #print(self)
    


    ## debug
    def class_sizes(self):
        sizes = []
        for c in self.keys(): sizes.append(len(self[c]))
        
        return sizes.copy()
