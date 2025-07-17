import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Callable, Type, Union # For GalaxySimulator type hints

# Assuming scoresbibm.tasks.base_task is available in your Python path
from scoresbibm.tasks.base_task import InferenceTask 

# --- Helper Class for Prior ---
class GalaxyPrior:
    """
    A simple prior distribution handler for galaxy parameters.
    Samples uniformly from ranges specified for each parameter.
    """
    def __init__(self, prior_ranges: Dict[str, Tuple[float, float]], param_order: List[str]):
        self.prior_ranges = prior_ranges
        self.param_order = param_order
        self.theta_dim = len(param_order)
        
        self.distributions = []
        for param_name in self.param_order:
            low, high = self.prior_ranges[param_name]
            self.distributions.append(torch.distributions.Uniform(torch.tensor(float(low)), torch.tensor(float(high))))

    def sample(self, sample_shape: Tuple[int]) -> torch.Tensor:
        """
        Generates samples from the prior.
        Args:
            sample_shape: A tuple containing the number of samples, e.g., (num_samples,).
        Returns:
            A PyTorch tensor of shape (num_samples, theta_dim).
        """
        num_samples = sample_shape[0]
        samples_per_param = [dist.sample((num_samples, 1)) for dist in self.distributions]
        return torch.cat(samples_per_param, dim=1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of theta under the prior.
        Assumes theta is a batch of shape (num_samples, theta_dim).
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0) # Make it (1, theta_dim)
            
        log_probs_per_param = []
        for i, dist in enumerate(self.distributions):
            log_probs_per_param.append(dist.log_prob(theta[:, i]))
        
        # Sum log_probs for independent parameters
        return torch.sum(torch.stack(log_probs_per_param, dim=1), dim=1)


# --- Custom Simformer Task ---
class GalaxyPhotometryTask(InferenceTask):
    """
    A Simformer InferenceTask for the GalaxySimulator.
    """
    # These are based on your provided 'inputs' and 'filter_codes'
    _theta_dim: int 
    _x_dim: int

    def __init__(self, 
                 name: str = "galaxy_photometry_task", 
                 backend: str = "jax",
                 prior_dict: Dict[str, Tuple[float, float]] = None,
                 param_names_ordered: List[str] = None,
                 run_simulator_fn: Callable = None,
                 num_filters: int = None):
        super().__init__(name, backend)

        #if prior_dict is None or param_names_ordered is None or run_simulator_fn is None or num_filters is None:
        #    raise ValueError("prior_dict, param_names_ordered, run_simulator_fn, and num_filters must be provided.")
        if param_names_ordered is None:
            #print('Warning: param_names_ordered is None. The parameter order will not be functional until set.')
            param_names_ordered = []

        self.param_names_ordered = param_names_ordered

        self._theta_dim = len(param_names_ordered) if param_names_ordered is not None else 0
        if num_filters is None:
            #print('Warning: num_filters is None. The x_dim will be set to 0 until num_filters is provided.')
            num_filters = 0
        
        self._x_dim = num_filters 
        if prior_dict is not None:
            self.prior_dist = GalaxyPrior(prior_ranges=prior_dict, param_order=self.param_names_ordered)
        else:
            #print('Warning: prior_dict is None. The prior distribution will not be functional until set.')
            self.prior_dist = None

        if run_simulator_fn is not None:
            self.run_simulator_fn = run_simulator_fn # This is your 'run_simulator' function
        else:
            #print('Warning: run_simulator_fn is None. The simulator will not be functional until set.')
            self.run_simulator_fn = None

    def get_theta_dim(self) -> int:
        return self._theta_dim
    
    def get_x_dim(self) -> int:
        return self._x_dim

    def get_prior(self):
        """Returns the prior distribution object."""
        # Simformer might expect a specific type of distribution object based on its backend.
        # Our GalaxyPrior samples PyTorch tensors. This is common in sbibm.
        return self.prior_dist

    def get_simulator(self):
        """
        Returns a callable that takes a batch of thetas and returns a batch of xs.
        The provided run_simulator_fn processes one sample at a time and returns a torch tensor.
        This wrapper will handle batching.
        """
        def batched_simulator(thetas_batch_torch: torch.Tensor) -> torch.Tensor:
            # Your run_simulator_fn expects a single sample and returns (1, x_dim) tensor.
            # We'll loop and cat.
            xs_list = []
            for i in range(thetas_batch_torch.shape[0]):
                theta_sample_torch = thetas_batch_torch[i, :] 
                # run_simulator_fn expects numpy array if it's not a dict, 
                # and handles tensor conversion internally.
                # It returns a tensor of shape [1, num_filters].
                x_sample_torch = self.run_simulator_fn(theta_sample_torch, return_type='tensor')
                xs_list.append(x_sample_torch)
            return torch.cat(xs_list, dim=0) # Shape will be (num_samples, num_filters)
            
        return batched_simulator
    
    def get_data(self, num_samples: int, **kwargs) -> Dict[str, jnp.ndarray]:
        """Generates and returns a dictionary of {'theta': thetas, 'x': xs} as JAX arrays."""
        prior = self.get_prior()
        simulator = self.get_simulator() # This is our batched_simulator

        # Sample thetas (parameters) using the prior
        # GalaxyPrior.sample returns a PyTorch tensor
        thetas_torch = prior.sample((num_samples,))
        
        # Simulate xs (photometry) using the parameters
        # batched_simulator also returns a PyTorch tensor
        xs_torch = simulator(thetas_torch)

        if self.backend == "jax":
            thetas_out = jnp.array(thetas_torch.cpu().numpy())
            xs_out = jnp.array(xs_torch.cpu().numpy())
        elif self.backend == "numpy":
            thetas_out = thetas_torch.cpu().numpy()
            xs_out = xs_torch.cpu().numpy()
        else: # "torch" or other
            thetas_out = thetas_torch
            xs_out = xs_torch
            
        return {"theta": thetas_out, "x": xs_out}

    def get_node_id(self) -> jnp.ndarray:
        """Returns an array identifying the nodes (dimensions) of theta and x."""
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch": # Should align with SBIBMTask if that's a reference
            return torch.arange(dim)
        else: # JAX or numpy
            return jnp.arange(dim)

    def get_base_mask_fn(self) -> Callable:
        """Defines the base attention mask for the transformer."""
        theta_dim = self.get_theta_dim()
        x_dim = self.get_x_dim()

        # Parameters only attend to themselves (or causal if ordered)
        thetas_self_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        
        # Data can attend to previous/current data points (causal within x)
        # Or use jnp.ones if full self-attention within x is desired.
        xs_self_mask = jnp.tril(jnp.ones((x_dim, x_dim), dtype=jnp.bool_))
        
        # Data can attend to all parameters
        xs_attend_thetas_mask = jnp.ones((x_dim, theta_dim), dtype=jnp.bool_)
        
        # Parameters do not attend to data
        thetas_attend_xs_mask = jnp.zeros((theta_dim, x_dim), dtype=jnp.bool_)

        base_mask = jnp.block([
            [thetas_self_mask, thetas_attend_xs_mask],
            [xs_attend_thetas_mask, xs_self_mask]
        ])
        base_mask = base_mask.astype(jnp.bool_)
        

        def base_mask_fn(node_ids, node_meta_data):    
            print(node_ids, node_meta_data)
            
            # Handles potential permutation/subsetting of nodes
            return base_mask[jnp.ix_(node_ids, node_ids)]
        
        return base_mask_fn
