"""
Reinforcement Learning
"""
from os import system
from time import sleep
from random import seed, uniform
from gym import make
from numpy import zeros, argmax, max as npmax


def q_learning(streets):
    """
    First we need to train our model. At a high level,
    we'll train over 10,000 simulated taxi runs.
    """
    q_table = zeros([streets.observation_space.n, streets.action_space.n])
    detail = {
        "learning_rate": 0.1,
        "discount_factor": 0.6,
        "exploration": 0.1,
        "epochs": 10000,
    }
    for _ in range(detail["epochs"]):
        state = streets.reset()
        done = False
        while not done:
            random_value = uniform(0, 1)
            if random_value < detail["exploration"]:
                action = streets.action_space.sample()
            else:
                action = argmax(q_table[state])
            next_state, reward, done, _ = streets.step(action)
            prev_q = q_table[state, action]
            next_max_q = npmax(q_table[next_state])
            new_q = (1 - detail["learning_rate"]) * prev_q + detail["learning_rate"] * (
                reward + detail["discount_factor"] * next_max_q
            )
            q_table[state, action] = new_q
            state = next_state
    return q_table


def wrapper():
    """
    wrapper function
    """
    seed(1234)
    streets = make("Taxi-v3").env
    print(streets.render())
    initial_state = streets.encode(2, 3, 2, 0)
    streets.s = initial_state
    print(streets.render())
    print(streets.P[initial_state])

    q_table = q_learning(streets)
    print(q_table[initial_state])
    for tripnum in range(1, 11):
        state = streets.reset()
        done = False
        trip_length = 0
        while not done and trip_length < 25:
            action = argmax(q_table[state])
            next_state, _, done, _ = streets.step(action)
            system("cls")
            print("Trip number " + str(tripnum) + " Step " + str(trip_length))
            print(streets.render(mode="ansi"))
            sleep(0.5)
            state = next_state
            trip_length += 1
        sleep(2)


if __name__ == "__main__":
    wrapper()
