import numpy as np
import open3d as o3d
from model.option import gen_options
import ipdb
class GmmSample:
    def __init__(self, inlier_ratio, sigma, uniform_range,
                 num_samples):
        self.inlier_ratio = inlier_ratio
        self.sigma = sigma
        self.uniform_range = uniform_range
        self.num_samples = num_samples

        self.rng = np.random.default_rng()
        self.bernoulli = np.random.binomial(1, inlier_ratio, self.num_samples)
        self.gaussian = np.random.normal(scale=sigma, size=self.num_samples)
        self.uniform = np.random.uniform(low=-uniform_range, high=uniform_range,
                                         size=self.num_samples)

    def sample(self, center, direction ):
        """
        center:  N x 3
        direction: N x 3
        """
        #print("start gmm sample")
        assert center.shape[0] == self.num_samples

        inlier_outlier = np.random.binomial(1, self.inlier_ratio, self.num_samples)
        
        inlier_index = np.squeeze(np.argwhere(inlier_outlier > 0.5))
        outlier_index = np.squeeze(np.argwhere(inlier_outlier < 0.5))
        gaussian_noise = np.random.normal(scale=self.sigma, size=inlier_index.shape[0])
        uniform_noise = np.random.uniform(low=-self.uniform_range, high=self.uniform_range,
                                          size=outlier_index.shape[0])

        #ipdb.set_trace()
        center[inlier_index] += direction[inlier_index] * np.repeat(gaussian_noise[:, np.newaxis], 3, axis=1)
        center[outlier_index] += direction[outlier_index] * np.repeat(uniform_noise[:, np.newaxis], 3, axis=1)
        #print("finish gmm smaple")
        return center


    
