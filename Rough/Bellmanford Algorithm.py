import numpy as np

v = np.zeros((4, 4)) # state value, initially all 0 for non-terminal states
q = np.zeros((4, 16)) # initialize all the action values for each state is 0

# Interpret actions into code by considering how (x, y) index change
#  "down", "right", "up", "left"
#  action format: [y_change, x_change]
actions = [[1, 0], [0, 1], [-1, 0], [0, -1]] # fill in each element

valid_locs = [[0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2],
              [2, 3], [3, 0], [3, 3]]

# Terminal state values
v[0, 3] = 100
v[3, 0] = 10
v[3, 3] = -100

gamma = 1.0

# define reward
r_0, r_1, r_2 = 100, 50, 50

def generate_episode():
    episode = []
    start_state = [2, 0]  # starting state
    current_state = start_state
    while current_state not in [[0, 3], [3, 0], [3, 3]]:
        action = np.random.choice(range(len(actions)))
        next_state = [current_state[0] + actions[action][0], current_state[1] + actions[action][1]]
        if next_state in valid_locs:
            if next_state == [2, 0]:
                reward = r_0
            else:
                reward = r_1 if next_state in [[0, 3], [3, 0], [3, 3]] else r_2
            episode.append((current_state, action, reward))
            current_state = next_state
        else:
            episode.append((current_state, action, -0.3))
            current_state = [2, 0]
    return episode

for i in range(1000):
    print(f'**************** interation [{i}] ****************')
    # Generate an episode
    episode = generate_episode()

    # Update the state values using Monte Carlo
    g = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        g = gamma * g + reward
        if state not in [[0, 3], [3, 0], [3, 3]]:
            v[state[0], state[1]] += 0.01 * (g - v[state[0], state[1]])

    print(v)
