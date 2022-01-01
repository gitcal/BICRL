"""
Both the updates are minor and are not expected to
have any big effects on results.
Updates:
   (1) Previously get_z function was always using
       boltz_beta=1, now you can pass this as
       argument to function.
   (2) kl_div function now finds unique demos from
       all demos and works with them. This makes
       the estimate slightly better.
"""

from copy import deepcopy
import numpy as np
import mdp_worlds
from mdp_utils import calculate_q_values
import mdp_utils

def is_same(demo1, demo2):
    # returns True if demo1 and demo2 are identical
    if len(demo1) != len(demo2):
        return False
    for (s1,a1), (s2,a2) in zip(demo1, demo2):
        if (not s1 == s2) or (not a1 == a2):
            return False
    return True


def get_z(env, boltz_beta, num_samples=100, use_unique=False):
        """
        Estimate the partition funciton Z empirically.
        """
	# Empirically estimates Z
        demos, demos_rewards,  Z = [], [], 0
        boltz_beta = boltz_beta
        bottom_right_corner = 8*6-1
        count = 0
        while count < num_samples:
            new_demo = mdp_utils.generate_boltzman_demo(env, boltz_beta, bottom_right_corner)
            add_to_demos = True
            if use_unique:
                for demo in demos:
                    if is_same(new_demo, demo):
                        add_to_demos = False
                        break
            if add_to_demos:
                count += 1
                demos.append(new_demo)
                demo_reward = 0
                for s, a in new_demo:
                    demo_reward += env.rewards[s]
                demo_reward = np.exp(demo_reward)
                demos_rewards.append(demo_reward)
                Z += demo_reward
        # sanity check that the Z gives a probability distribution
        assert(np.abs(np.sum(np.array(demos_rewards)/Z) - 1) < 0.05)
        return Z

def kl_div(demos, mdp, Z):
    """
    This function assumes that all demos were sampled from a uniform
    distribution; it then calculates the probabilities of these demos
    under the given MDP and calculates KL.
    demos: Expert demonstrations given to the algo.
    """
    kl = 0
    demos = deepcopy(demos)  # just for safety
    num_demos = len(demos)

    # Find unique demos
    p = np.zeros((num_demos, num_demos))
    for i in range(num_demos):
        for j in range(num_demos):
            p[i][j] += int(is_same(demos[i], demos[j]))
    probs = np.sum(p, axis=0).tolist()
    unique_demos = []
    unique_demos_probs = []
    for k, demo in zip(probs, demos):
        if k > 1:
            demo_added = True
            for added_demo in unique_demos:
                if is_same(demo, added_demo):
                    demo_added = False
            if demo_added:
                unique_demos.append(demo)
                unique_demos_probs.append(k)
        else:
            unique_demos.append(demo)
            unique_demos_probs.append(k)

    for p, demo in zip(unique_demos_probs, unique_demos):
        R = 0
        for s, a in demo:
            R += mdp.rewards[s]
        q = np.exp(R)/	Z
        kl += (p + np.log(p/q))
    return kl


class scobee:
    def __init__(self, env, demos, boltz_beta=1, epsilon=0.0001):
        self.env       = env
        self.demos     = demos
        self.boltz_beta= boltz_beta
        self.epsilon   = epsilon
        self.converged = False
        self.old_kl    = 0

    def run_one_more_iter(self):
        """
        This is an inefficient implementation of Scobee's algorithm.
        ---- Start With Empty Constraint Set
        while states still left to be constrained:
                ---- Run Q-value
                ---- Constrain the state with highest Q value and not
                         part of the demos.
                ---- Add that state to constrained
        """
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        # argsort the max q values in each state
        q_values_max = np.max(q_values, axis=1)
        q_values_argsort = np.flip(q_values_max.argsort())
        # q_values_argsort[0] contains idx of state with highest q value
        for i in q_values_argsort:
            # check if i is in demos
            if (not self.is_in_demos(i) and 
                    i not in self.env.terminals and
                    self.env.constraints[i] != 1):
                self.env.constraints[i] = 1
                # also modify the rewards for that state
                self.env.rewards[i]     = -10
                break
        else:
            print("Found all possible constraints already!")
            self.converged = True
        self.new_kl = kl_div(self.demos, self.env, get_z(self.env, self.boltz_beta))
        delta_kl    = self.new_kl - self.old_kl
        self.old_kl = self.new_kl
        print(delta_kl)

    def find_k_constraints(self, k):
        for i in range(k):
            self.run_one_more_iter()

    def find_all_constraints(self):
        while not self.converged:
            self.run_one_more_iter()

    def is_in_demos(self, state):
        for demo in self.demos:
            for s, a in demo:
                if state == s:
                    return True
        return False

if __name__ == '__main__':
    nx = 8 # columns
    ny = 6 # rows
    gamma = 0.95
    noise = 0.1
    terminal_states = [0]#[rows * columns - 1]
    # rewards negative everywhere except for the goal state
    rewards = - np.ones(ny * nx)
    rewards[0] = 2.0
    constraints = np.zeros(ny * nx)
    # set some constrains by hand
    constraints[[16, 17, 24, 25, 23, 31]] = 1
    rewards[[16, 17, 24, 25, 23, 31]] = -10
    num_cnstr = len(np.nonzero(constraints)[0])
    env = mdp_worlds.nonrand_gridworld(ny, nx, terminal_states, rewards, constraints, gamma, noise)

    bottom_right_corner = ny * nx - 1
    trajectory_demos = []
    boltz_beta = 10000.0
    num_of_demos = 20
    for i in range(num_of_demos):
        # trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_left_corner))
        trajectory_demos.append(mdp_utils.generate_boltzman_demo(env, boltz_beta, bottom_right_corner))
        #print(trajectory_demos[i])

    constraints = np.zeros(ny * nx)
    test_env = mdp_worlds.nonrand_gridworld(ny, nx, terminal_states, rewards, constraints, gamma, noise)
    # ============================================================
    cl = scobee(test_env, trajectory_demos, boltz_beta)
    cl.find_k_constraints(4)
    mdp_utils.print_array_as_grid(test_env.constraints, test_env)
    cl.find_all_constraints()
    # ============================================================
    print("True Constraints")
    mdp_utils.print_array_as_grid(env.constraints, test_env)
    print("=============================================")
    print("Constraints found by algo")
    mdp_utils.print_array_as_grid(test_env.constraints, test_env)
    print("=============================================")
    state_visited_by_expert_or_not = np.zeros(ny*nx)
    for s,a in trajectory_demos[0]:
    	state_visited_by_expert_or_not[s] = 1
    mdp_utils.print_array_as_grid(state_visited_by_expert_or_not, test_env)
