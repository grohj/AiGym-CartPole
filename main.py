import gym
import random
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.utils as utils
import time
import keras

env = gym.make('CartPole-v1')
env.reset()

size = 200000

goal_steps = 500
score_requirement = 20
intial_games = 10000

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log.tf")


def play_a_random_game_first():
    for step_index in range(goal_steps):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if reward != 0.0:
            print("Step {}:".format(step_index))
            print("action: {}".format(action))
            print("observation: {}".format(observation))
            print("reward: {}".format(reward))
            print("done: {}".format(done))
            print("info: {}".format(info))

    env.reset()


def model_data_preparation():
    dataX = []
    dataY = []
    accepted_scores = []
    for game_index in range(intial_games):  # 10000
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):  # 500
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                dataX.append(data[0])
                dataY.append(output)

        env.reset()
    return np.array(dataX), np.array(dataY)


def gatherData():
    dataX = []
    dataY = []
    while len(dataX) < size:
        print(len(dataX))
        currX = []
        currY = []
        currScore = 0
        done = False
        while currScore < score_requirement or not done:

            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)
            # env.render()

            currScore += reward
            currX.append(observation)
            arr = [0.0, 0.0]
            arr[action] = 1.0
            currY.append(arr)

            if done:
                env.reset()

        if currScore >= score_requirement:
            dataX += currX
            dataY += currY
            env.reset()

    print("end")
    return np.array(dataX), np.array(dataY)


def createModel():
    model = Sequential()
    model.add(Dense(4, activation="relu", input_shape=(4,)))
    model.add(Dense(64, activation="sigmoid", input_shape=(4,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss='mse', metrics=["accuracy"])
    return model


def playAfterTrain():
    env.reset()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    while True:
        env.render()
        input = np.array([observation])
        pred = model.predict(input)
        decision = np.argmax(pred)
        observation, reward, done, info = env.step(decision)

        if done:
            env.reset()
            # time.sleep(1)


# play_a_random_game_first()
dataX, dataY = model_data_preparation()
model = createModel()
model.summary()

model.fit(dataX, dataY, epochs=10, callbacks=[tensorboard_callback])
playAfterTrain()
