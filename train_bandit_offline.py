import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt
from tqdm import tqdm


def reward_linear(a):
    if a > -0.25 and a < 0.25:
        return np.abs(a)-0.125
    else:
        return float('-inf')

class SimpleSpecs():
    """SimpleSpecs class to define the action space of the bandit problem. 
        This would usually be returned by the Gym environment but for this simple example we define it here.
    """
    def __init__(self, shape, dtype, minimum, maximum):
        self.shape = shape
        self.dtype = dtype
        self.minimum = minimum
        self.maximum = maximum
    
def sample_reward(rew_env, act_policy, n_samples):
    actions=act_policy(n_samples)
    rewards=np.array([rew_env(action) for action in actions])

    return rewards, actions

class LaplacePDF:
    """Laplace distribution necessary to fit the behavioural policy $\mu(a)$.

    The Laplace distribution has two parameters, location (mu - center) and scale (b - diversity).
    The equation describing it is $f(x | \mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)$.
    """
    def __init__(self, loc=0, scale=1):
        """Initializes the Laplace distribution.
        Args:
            loc: location of the center, default 0.
            scale: scale of the diversity (greater than 0), default 1.
        """
        self.loc = loc
        self.scale = scale
    def __call__(self, x):
        """Calculate the probability density function value at a point.

        Args:
            x: point at which to calculate the probability density function.
        """
        # return (np.exp(-abs(x - self.loc) / self.scale) / (2 * self.scale))
        return laplace.pdf(x, loc=self.loc, scale=self.scale)

class UniformPolicy:
    """Uniform policy from which we will sample the actions for the environment between low and high.
    """
    def __init__(self, low=0, high=1):
        self.low=low
        self.high=high
    def __call__(self, n_samples=1):
        return np.random.uniform(self.low, self.high, n_samples)
    
import tensorflow as tf

# Structure of the Reward net for BRAC. Since for BRAC we only need to train the reward function, we only need a reward net.
class RewardNet(tf.keras.Model):
  def __init__(self,
               action_dim,
               hidden_dims=(32, 32)):
    """Creates a neural net.

    Args:
      action_dim: Action size.
      hidden_dims: List of hidden dimensions.
    """
    super(RewardNet, self).__init__()
    # Define the initializers like in fisher_brc/critic.py
    # This is the critical part: without this initializers I had convergence problems.
    relu_orthogonal_init = tf.keras.initializers.Orthogonal(gain=tf.math.sqrt(2.0))
    near_zero_orthogonal_init = tf.keras.initializers.Orthogonal(1e-2)

    # Define the input layer
    inputs = tf.keras.Input(shape=(action_dim,))

    # Define hidden layers
    layers = [] 
    for hd in hidden_dims:
        layers.append( 
            tf.keras.layers.Dense(
                hd,
                activation=tf.nn.relu,
                kernel_initializer=relu_orthogonal_init
            )
        )

    # Define the output as the sequential layers and final output combination.
    # The output is always one-dimensional as this network is for a reward function
    # The outputs sequential is called with the input layer.
    outputs = tf.keras.Sequential(
        layers + [ tf.keras.layers.Dense(1, kernel_initializer=near_zero_orthogonal_init) ]
    )(inputs)
    
    # Save the net
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # End of model definition

  # Define the call function with the tf.function decorator to "make it faster"
  # It becomes a graph and is optimized by TensorFlow
  @tf.function
  def call(self, actions):
    """Returns R-value estimates for given actions.

    Args:
      actions: A batch of actions.

    Returns:
      The estimate of R-values.
    """
    return tf.squeeze(self.model(actions), 1)
  
# Training loop
def train_simple_brac(actions, rewards, num_epochs):
    optimizer = tf.keras.optimizers.Adam()
    loss_fn_brac = tf.keras.losses.MeanSquaredError()

    reward_net = RewardNet(action_dim=1, hidden_dims=(32, 32))

    for epoch in tqdm(range(num_epochs)):
        with tf.GradientTape() as tape:
            predicted_rewards = reward_net(actions)
            loss = loss_fn_brac(rewards, predicted_rewards)

        gradients = tape.gradient(loss, reward_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, reward_net.trainable_variables))
        
        if epoch % (n_epochs // 10) == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')
            for var, grad in zip(reward_net.trainable_variables, gradients):
                print(f'{var.name}, Gradient: {tf.reduce_mean(grad).numpy()}')

    return reward_net 

import tensorflow_probability as tfp

tfd = tfp.distributions
LOG_STD_MIN = -20
LOG_STD_MAX = 2
class SimplifiedPolicy(tf.keras.Model):
    def __init__(self, state_dim, action_spec, hidden_dims=(32, 32)):
        """
        Simplified version of a policy for the Fisher-BRC actor.

        Args:
            - state_dim (int): Dimension of the state space.
            - action_spec (SimpleSpecs): Specifications of the action space.
            - hidden_dims (tuple): Tuple of hidden dimensions for the policy network.
        """
        super(SimplifiedPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_spec = action_spec
        self.action_mean = tf.constant((action_spec.maximum + action_spec.minimum) / 2.0, dtype=tf.float32)
        self.action_scale = tf.constant((action_spec.maximum - action_spec.minimum) / 2.0, dtype=tf.float32)
        self.action_dim = action_spec.shape[0]
        action_dim = self.action_dim

        self.hidden_layers = [tf.keras.layers.Dense(dim, 
                                                    activation='relu', 
                                                    kernel_initializer=tf.keras.initializers.Orthogonal(tf.math.sqrt(2.0))
                                                ) for dim in hidden_dims]
        
        if self.state_dim == 0:
            inputs = tf.keras.Input(shape=(1,)) # this should be the state_dim but I can't avoid it and I will use a uniform distribution putting in data
        else:
            inputs = tf.keras.Input(shape=(self.state_dim,))
        outputs = tf.keras.Sequential(
            self.hidden_layers + [tf.keras.layers.Dense(
                action_dim*2, # double because I need to split it in mean and log_std
                kernel_initializer=tf.keras.initializers.Orthogonal(1e-2))]
        )(inputs)
        self.trunk = tf.keras.Model(inputs=inputs, outputs=outputs) 
        # The trunk inputed with n values will return couple of values that will be use as the mean and standard
        # deviation of the Gaussian distribution.

    def __get_dist(self, states, stddev=1.0):
        """From the states, returns a distribution of actions.
        """
        if states is None and self.state_dim == 0:
            # For this simplified version, we'll ignore states
            states = tf.random.uniform((1,), minval=-self.action_spec.minimum, maxval=self.action_spec.maximum)
        out = self.trunk(states)
        
        mean, log_std = tf.split(out, num_or_size_splits=2, axis=1)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        dist = tfd.TransformedDistribution(
            tfd.Normal(loc=mean, scale=std*stddev),
            tfp.bijectors.Chain([
                tfp.bijectors.Shift(shift=self.action_mean),
                tfp.bijectors.Scale(scale=self.action_scale),
                tfp.bijectors.Tanh(),
            ])
        )

        return dist

    @tf.function   
    def call(self, states, sample=False, with_log_probs=False):
        
        if sample:
           dist = self.__get_dist(states)
        else:
            dist = self.__get_dist(states, stddev=0.0)
        actions = dist.sample()

        # actions = tf.clip_by_value(actions, self.action_spec.minimum, self.action_spec.maximum)
    
        if with_log_probs:
            log_probs = dist.log_prob(actions)
            return actions, log_probs
        else:
            return actions


# Structure of the Fisher BRC agent (actor-critic)
# We could use the FBRC object from the paper, but that uses Behavioural cloning for the behavioural policy.
# Unfortunately for the simple example we need a Laplacian (or Gaussian) fitted policy.
# It also implements soft actor critic which is unneccesary complicate.
class FisherBRC(object):
    """
    The Fisher BRC needs:
        - actor NN (+ optimizer)
        - behavioural policy 
        - one critic(offset) to estimate the values of the Offset function $O_\theta$) (+optimizer)
       
    It also has the following tunable parameters:
        - actor learning rate for Adam optimizer
        - alpha parameter controlling the entropy in the actor loss
        - critic/offset learnign rate for Adam optimizer
        - lambda: regularizaion parameter of eq 7 of the paper

    However, differently from the original work here I re-implement stuff because:
        - we don't need double critics as the problem is simple. 
        - we don't need the actor to be a policy network, as for the bandit example a uniform policy is computed.
        - no observations space is needed.
        - behavioural policy is a Laplacian (or Gaussian) fitted policy, not a mixture of Gaussian.
    """
    def __init__(self,
                 observation_spec = SimpleSpecs(shape=(0,), dtype=tf.float32, minimum=-1, maximum=1),
                 action_spec = SimpleSpecs(shape=(1,), dtype=tf.float32, minimum=-1, maximum=1),
                 behavioural_policy = None,
                 hiddim__actor = (32, 32),
                 lr__actor = 3e-4,
                 alpha = 0.001,
                 hiddim__offset=(32, 32),
                 lr__offset = 3e-4,
                 regularizer = 0.1): #lambda
        """Creates a Fisher BRC agent. (Creates the elements plus saves the parameters).

        Args:
            - state_dim (int): Dimension of the state space.
            - action_dim (int): Dimension of the action space.
            - behavioural_policy (callable): Behavioural policy to sample actions from.
            - hiddim__actor (tuple): Tuple of hidden dimensions for the actor network.
            - lr__actor (float): Learning rate for the actor network.
            - alpha (float): Value of the entropy coefficient.
            - hiddim__offset (tuple): Tuple of hidden dimensions for the offset network.
            - lr__offset (float): Learning rate for the offset network.
            - regularizer (float): Regularization parameter for the offset network.
        """
        assert behavioural_policy is not None, "Behavioural policy must be defined."
        self.behavioural_policy = behavioural_policy

        self.actor = SimplifiedPolicy(state_dim=observation_spec.shape[0], 
                                      action_spec=action_spec, 
                                      hidden_dims=hiddim__actor)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr__actor)
        self.alpha = alpha
        
        self.offset = RewardNet(action_dim=action_spec.shape[0],
                                hidden_dims=hiddim__offset) 
        self.offset_optimizer = tf.keras.optimizers.Adam(learning_rate=lr__offset)
        self.regularizer = regularizer

    def q_value(self, actions, stop_gradient=False):
        """Function to estimate the Q-value of the critic. In Fisher BRC this is defined as the 
        offset plus log of the behaviour policy. This function calculates this sum.
           
        This function does what dist_critic does in the original code."""

        # Calculate the offset and the log of behavioural policy
        offset = tf.cast(self.offset(actions), tf.float32)
        log_pi = tf.cast(tf.math.log(self.behavioural_policy(actions)), tf.float32)

        # For the computation of some gradients we don't need the log_pi to be considered so we prevent
        # their contribution. This function is necessary for the building of the graph and stuff.
        if stop_gradient:
            log_pi = tf.stop_gradient(log_pi)

        return (offset + log_pi) # q_value = offset + log_pi
    
    def fit_critic(self, states, actions, nxt_states,
                         rewards, discounts):
        """Updates critic parameters.

        Args:
            states: Batch of states.
            actions: Batch of actions.
            next_states: Batch of next states.
            rewards: Batch of rewards.
            discounts: Batch of masks indicating the end of the episodes.

        Returns:
            Dictionary with information to track.
        """
        # Sample actions from the policy what the actor would take at current state
        policy_actions = self.actor(states=states, sample=True)
        
        offset_variables = self.offset.trainable_variables

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(offset_variables)

            q = self.q_value(actions, stop_gradient=True)

            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as inner_tape:
                inner_tape.watch([policy_actions])

                # Calculate the current offset (for Expected value of squared gradients in eq 7, i.e., after lambda)
                o_reg = self.offset(policy_actions)

            # Calculate the squared value of the norm of the gradient of the offset (eq 7)
            o_grads = inner_tape.gradient(o_reg, policy_actions)
            o_grads_norm = tf.reduce_sum(tf.square(o_grads), axis=-1)

            del inner_tape

            # Calculate the loss of the critic (eq 7 + )
            J = tf.losses.mean_squared_error(rewards, q) # eq 2 of the paper
            critic_loss = J + self.regularizer * o_grads_norm

        # Compute the gradients of the critic loss and updated the offset params
        critic_grads = tape.gradient(critic_loss, offset_variables)
        self.offset_optimizer.apply_gradients(zip(critic_grads, offset_variables))

        return {
            'q': tf.reduce_mean(q),
            'critic_loss': critic_loss,
            # 'q_grad': # tf.reduce_mean(critic_grads)
        }

    def fit_actor(self, states):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor.trainable_variables)
            actions, log_probs = self.actor(states, sample=True, with_log_probs=True)
            q = self.q_value(actions) # get what is the critic opinion for that random action
            actor_loss = tf.reduce_mean(self.alpha*log_probs - q) # -1 * alpha * log_probs == alpha * entropy 

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return {
            'actor_loss': actor_loss
        }
    
    def update_step(self, actions, rewards):
        """Performs a single training step for critic (and actor).

        Args:
            actions and rewards for a simple toy bandit problem

        Returns:
            Dictionary with losses to track.
        """
        critic_dict = self.fit_critic(states=None, actions=actions, 
                                      nxt_states=None, 
                                      rewards=rewards, 
                                      discounts=1.0) # I put one for the discounts as there is no final state and I guess I should go on and on

        actor_dict = self.fit_actor(None)

        return {**actor_dict, **critic_dict}

def train_simple_fbrc(action, rewards, n_epochs, reg_lambda):
    # Create the Fisher BRC agent
    fisher_brc = FisherBRC(observation_spec=SimpleSpecs(shape=(0,), dtype=tf.float32, minimum=-1, maximum=1), # No states
                            action_spec=SimpleSpecs(shape=(1,), dtype=tf.float32, minimum=-1, maximum=1),
                            behavioural_policy=laplace_pdf,
                            hiddim__actor=(32, 32),
                            lr__actor=3e-4,
                            alpha=0.001,
                            hiddim__offset=(32, 32),
                            lr__offset=3e-4,
                            regularizer=reg_lambda)
    
    for epoch in tqdm(range(n_epochs)):
        report = fisher_brc.update_step(action, rewards)
        
        if epoch % (n_epochs // 10) == 0:
            print(f'Epoch {epoch}, Cr-Loss: {report["critic_loss"].numpy()}, Actor-Loss: {report["actor_loss"].numpy()}')

    return fisher_brc


if __name__ == "__main__":
    
    rew_env = lambda a: reward_linear(a)
    
    # From paper: The sampling policy is a uniform but limited between -0.25 and 0.25 
    act_policy = UniformPolicy(-0.25, 0.25)
    
    # From paper: Sample 1000 rewards
    n_samples = 1000
    rewards, actions = sample_reward(rew_env, act_policy, n_samples)

    # From paper: Fit the behaviour policy to the reward environment as laplace distribution
    # Estimate parameters using Maximum Likelihood Estimation from SciPy
    mu, b = laplace.fit(actions)
    laplace_pdf = LaplacePDF(mu, b)

    # Fit the BRAC critic
    n_epochs=1000
    trained__brac=train_simple_brac(actions, rewards, n_epochs)

    # Plot the results
    x=np.linspace(-1, 1, 1000)
    orig_fig=plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for alpha in [0.01, 0.1, 1.0]:
        y=np.array(trained__brac(x)).reshape(-1) # the output of reward_net has shape (1000, 1) while numpy stuff is (1000,)
        y=y+alpha*np.log(laplace_pdf(x))
        plt.scatter(x, y,label=f'alpha={alpha:.2f}')
    
    plt.title('BRAC')
    plt.xlabel('Actions')
    plt.ylabel('Rewards')
    plt.xlim(-1, 1)
    plt.ylim(-2, 2)
    plt.scatter(actions, rewards, color='r', marker='.', label='Data')
    plt.grid(visible=True)
    plt.legend()
    
    # Train  FBRC and plot the results (add also \lambda=0 to make sure the network learns correctly)
    n_epochs=1000
    trained__fbrc = train_simple_fbrc(actions, rewards, n_epochs, 0)
    y=np.array(trained__fbrc.q_value(x)).reshape(-1)

    fbrc_ax = plt.subplot(1, 2, 2)
    plt.scatter(x, y,label=f'lambda=0')
    n_epochs=10000
    for lambda_exp in range(-1, 0, 1):
        lmbd = lambda e: 10**e
        print(f'Training for lambda {10**lambda_exp}')
        trained__fbrc = train_simple_fbrc(actions, rewards, n_epochs, 10**lambda_exp)
        y=np.array(trained__fbrc.q_value(x)).reshape(-1)
        plt.sca(fbrc_ax)
        plt.scatter(x, y,label=f'lambda={10**lambda_exp}')

    plt.title('F-BRC')
    plt.xlabel('Actions')
    plt.ylabel('Rewards')
    plt.xlim(-1, 1)
    plt.ylim(-2, 2)
    plt.scatter(actions, rewards, color='r', marker='.', label='Data')
    plt.grid(visible=True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('figure_1.png')