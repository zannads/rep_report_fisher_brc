import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt
from tqdm import tqdm


def reward_linear(a):
    if a > -0.25 and a < 0.25:
        return np.abs(a)-0.125
    else:
        return float('-inf')
    
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
    # Define the loss function for the BRAC
    optimizer = tf.keras.optimizers.Adam()
    loss_fn_brac = tf.keras.losses.MeanSquaredError()

    # Create the reward network
    reward_net = RewardNet(1)

    # Train for multiple epochs
    for epoch in tqdm(range(num_epochs)):
        with tf.GradientTape() as tape:
            # Predict reward using the model
            predicted_rewards = reward_net(actions)
            # Calculate mean squared error loss
            loss = loss_fn_brac(rewards, predicted_rewards)

        # Compute gradients and update model parameters
        gradients = tape.gradient(loss, reward_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, reward_net.trainable_variables))
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')
            for var, grad in zip(reward_net.trainable_variables, gradients):
                print(f'{var.name}, Gradient: {tf.reduce_mean(grad).numpy()}')

    return reward_net  # Return the trained reward network


# Soft update function necessary because eq 2 of the paper uses a network and target network to compute the loss
# def soft_update(net, target_net, tau=0):
#     """Soft update rule for the target network. It is used to update the target network with the parameters of the main network.
    
#     Args:
#         - net: main network (usually updated with the optimizer)
#         - target_net: target network (updated with this method)
#         - tau: soft update rule parameter (default tau=0 means no update. tau=1 means copy the parameters from the main net)
#     """
#     for var, target_var in zip(net.variables, target_net.variables):
#         target_var.assign(tau * var + (1 - tau) * target_var)

# Structure of the Fisher BRC agent (actor-critic) + training performing.
# We could use the FBRC object from the paper, but that uses Behavioural cloning for the behavioural policy.
# Unfortunately for the simple example we need a Laplacian (or Gaussian) fitted policy.
class FisherBRC(object):
    """
    The Fisher BRC needs:
        - actor NN (+ optimizer)
        - behavioural policy 
        - one critic(offset) to estimate the values of the Offset function $O_\theta$) (+optimizer)
        - one critic target(offset target) to estimate J (eq 2 of the paper) to calculate the loss (no optimizer, updated with soft update rule).
        
    It also has the following tunable parameters:
        - actor learning rate for Adam optimizer
        - alpha: the temperature learning rate for the actor entropy learning
        - alpha learning rate for adam optimizer, because log(alpha) can also be learned at runtime
        - target_entropy for the actor 

        - critic learnign rate for Adam optimizer
        - tau for the soft update rule of the critic target

        - gamma: discount of the environment
        - lambda: regularizaion parameter of eq 7 of the paper
        - reward bonus for some reason (appears in code but reason behind it is not clear why)

    However, differently from the original work here I re-implement stuff because:
        - we don't need double critics as the problem is simple. 
        - we don't need the actor to be a policy network, as for the bandit example a uniform policy is computed.
        - no observations space is needed.
        - behavioural policy is a Laplacian (or Gaussian) fitted policy, not a mixture of Gaussian.
    """
    def __init__(self,
                 state_dim = 1,
                 action_dim = 1,
                 offset_hidden_dims=(32, 32),
                 lr__offset = 3e-4,
                 behavioural_policy = None,
                 # discount = 0.99,
                 # tau = 0.005,
                 regularizer = 0.1,
                 reward_bonus = 0.0):
        """Creates a Fisher BRC agent. (Creates the elements plus saves the parameters).

        Args:
            - stated_dim (int): State size
            - action_dim (int): Action size
            - offset_hidden_dims (tuple): Tuple of hidden dimensions for the offset network
            - lr__offset (float): Learning rate for the Adam optimizer of the offset network
            - discount (float): Discount factor
            - tau (float): Soft update rule parameter
            - regularizer (float): Regularization parameter of equation 7 of the paper (lambda)
            - reward_bonus (float): Reward bonus for some reason they use it in the code
        """
        self.actor = UniformPolicy(-1, 1)
        
        # In the toy example we only need one critic and there is no state so let's just reuse the RewardNet and drop state_dim
        self.offset = RewardNet(action_dim, hidden_dims=offset_hidden_dims) 
        self.offset_optimizer = tf.keras.optimizers.Adam(learning_rate=lr__offset)

        # self.offset_target = RewardNet(action_dim, hidden_dims=offset_hidden_dims) # exactly the same structure as the offset net
        #update the parameters of the offset network so that they start from the same point
        
        self.behavioural_policy = behavioural_policy

        # self.discount = discount
        # self.tau = tau
        self.regularizer = regularizer
        self.reward_bonus = reward_bonus

    def q_value(self, actions, stop_gradient=False):
        """Function to estimate the Q-value of the critic. In Fisher BRC this is defined as the 
        offset plus log of the behaviour policy. This function calculates this sum.
        The use target net is necessary because in equation 2 a target net is used to calculate the loss.
            
        This function does what dist_critic does in the original code."""

        # Calculate the offset and the log of behavioural policy
        offset = tf.cast(self.offset(actions), tf.float64)
        log_pi = tf.cast(tf.math.log(self.behavioural_policy(actions)), tf.float64)

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
        offset_variables = self.offset.trainable_variables
        # Sample actions from the policy what the actor would take at current state
        # Convert numpy arrays to tensors because differently from the orginal implementation I don't use tf probabilites 
        policy_actions = tf.convert_to_tensor(self.actor(n_samples=len(actions)), dtype=tf.float64)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(offset_variables)

            # Calculate the current Q=Q_net(s,a). (for eq 2 of the paper - for J in eq 7)
            q = self.q_value(actions, stop_gradient=True)

            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as inner_tape:
                inner_tape.watch([policy_actions])

                # Calculate the current offset (for Expected value of squared gradients in eq 7, i.e., after lambda)
                offset_reg = self.offset(policy_actions)

            # Calculate the squared value of the norm of the gradient of the offset (eq 7)
            offset_grads = inner_tape.gradient(offset_reg, policy_actions)
            offset_grad_norm = tf.reduce_sum(tf.square(offset_grads), axis=-1)

            del inner_tape

            # Calculate the loss of the critic (eq 7 + )
            J = tf.losses.mean_squared_error(rewards, q) # eq 2 of the paper
            critic_loss = J + self.regularizer * offset_grad_norm

        # Compute the gradients of the critic loss and updated the offset params
        critic_grads = tape.gradient(critic_loss, offset_variables)
        self.offset_optimizer.apply_gradients(zip(critic_grads, offset_variables))

        # Update the target critic with the soft update rule
        # soft_update(self.offset, self.offset_target, tau=self.tau)

        return {
            'q': tf.reduce_mean(q),
            'critic_loss': critic_loss,
            # 'q_grad': # tf.reduce_mean(critic_grads)
        }
    
    def fit_actor(self, states):
       pass
        
    def update_step(self, actions, rewards):
        """Performs a single training step for critic (and actor).

        Args:
        actions and rewards for a simple toy bandit problem

        Returns:
        Dictionary with losses to track.
        """
        int_rewards = rewards + self.reward_bonus

        critic_dict = self.fit_critic(None, actions, None, int_rewards, 1.0) # I put one for the discounts as there is no final state and I guess I should go on and on

        actor_dict = self.fit_actor(None)

        return critic_dict

def train_simple_fbrc(action, rewards, n_epochs, reg_lambda):
    # Create the Fisher BRC agent
    fisher_brc = FisherBRC(state_dim=0, action_dim=1, 
                           behavioural_policy = laplace_pdf,
                           offset_hidden_dims=(256, 256, 256),
                           #discount = 0.9,
                           # tau = 0.01,
                           lr__offset=3e-4,
                           regularizer = reg_lambda,
                           reward_bonus = rew_bonus_fbrc )
    
    # Train for multiple epochs
    for epoch in tqdm(range(n_epochs)):
        # Perform a single training step
        report = fisher_brc.update_step(action, rewards)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {report["critic_loss"].numpy()}')

    return fisher_brc # Return the trained actor-critic agent 


if __name__ == "__main__":
    
    # Define the reward environment
    rew_env = lambda a: reward_linear(a)
    
    # The sampling policy is a uniform but limited between -0.25 and 0.25 
    act_policy = UniformPolicy(-0.25, 0.25)
    
    # Sample rewards
    n_samples = 1000
    rewards, actions = sample_reward(rew_env, act_policy, n_samples)

    # Fit the behaviour policy to the reward environment (needed for both BRAC and F-BRC)
    # Estimate parameters using Maximum Likelihood Estimation from SciPy
    mu, b = laplace.fit(actions)
    laplace_pdf = LaplacePDF(mu, b)
    print(f"Behaviour policy fitted. mu: {mu}, b: {b}")

    plt.figure(figsize=(10, 5))
    plt.title('Fitted Behaviour Policy')
    """
    plt.scatter(np.linspace(-1, 1, 1000), laplace_pdf(np.linspace(-1, 1, 1000)), color='r', marker='.', label='Fitted Laplace')
    plt.scatter(np.linspace(-1, 1, 1000), np.log(laplace_pdf(np.linspace(-1, 1, 1000))), color='b', marker='.', label='log(Fitted Laplace)')
    plt.savefig('behaviour_policy.png')
    """
    # Fit the BRAC critic
    n_epochs=1000
    trained__brac=train_simple_brac(actions, rewards, n_epochs)

    # Plot the results
    x=np.linspace(-1, 1, 1000)
    plt.figure(figsize=(10, 5))
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
    plt.legend()

    rew_bonus_fbrc = np.min(-0.125-np.log(laplace_pdf(actions)))
    plt.subplot(1, 2, 2)
    for lambda_exp in range(-3, 2, 1):
        lmbd = lambda e: 10**e
        print(f'Training for lambda {10**lambda_exp}')
        trained__fbrc = train_simple_fbrc(actions, rewards, n_epochs, 10**lambda_exp)
        y=np.array(trained__fbrc.q_value(x)).reshape(-1)-rew_bonus_fbrc
        # no need to add log of mu as it is done inside q_value method
        plt.scatter(x, y,label=f'lambda={10**lambda_exp}')
    trained__fbrc = train_simple_fbrc(actions, rewards, n_epochs, 0)
    y=np.array(trained__fbrc.q_value(x)).reshape(-1)-rew_bonus_fbrc
    plt.scatter(x, y,label=f'lambda=0')

    plt.title('F-BRC')
    plt.xlabel('Actions')
    plt.ylabel('Rewards')
    plt.xlim(-1, 1)
    plt.ylim(-2, 2)
    plt.scatter(actions, rewards, color='r', marker='.', label='Data')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figure_1.png')