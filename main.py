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


goal_steps = 500
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

def gatherData(count, minimumScore):
    dataX = []
    dataY = []
    while len(dataX) < count:
        print(len(dataX))
        currX = []
        currY = []
        currScore = 0
        env.reset()
        observation = []
        done = False
        while not done:
            action = random.randrange(0, 2)
            if observation != []:
                currX.append(observation)
                arr = [0.0, 0.0]
                arr[action] = 1.0
                currY.append(arr)
            observation, reward, done, info = env.step(action)
            # env.render()

            currScore += reward



        if currScore >= minimumScore:
            dataX += currX
            dataY += currY

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
dataX, dataY = gatherData(200000, 40)
model = createModel()
model.summary()

model.fit(dataX, dataY, epochs=10, callbacks=[tensorboard_callback])
playAfterTrain()
