from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from math import floor


class Agent:
    def __init__(self, state_size, action_size, model_name=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.model_name = model_name
        self.gamma = 0.95
        self.data = 10_000_000
        self.epsilon = .70
        self.epsilon_min = .1
        self.epsilon_decay = float(np.e)**float(np.log(self.epsilon_min/self.epsilon)/self.data)
        print('stat:', self.data, self.epsilon_decay)
        if model_name:
            try:
                print('loading model')
                self.model = load_model(f'model/{model_name}')
            except:
                print('fail to load model, creating new model')
                self.model = self.model()
        else:
            print('creating new model')
            self.model = self.model()

    def model(self):
        model = Sequential()
        model.add(Dense(units=4096, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(units=4096, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if random.random() <= self.epsilon:
                possible_action = []
                current_loc = None
                body = []
                for e, i in enumerate(state[0]):
                    if i == -1:
                        current_loc = e
                    elif i != 0 and i != 1:
                        body.append(e)
                row = floor(current_loc / 7)
                for action, location in enumerate([current_loc - 7, current_loc - 1, current_loc + 7, current_loc + 1]):
                    diff = abs(location - current_loc)
                    if not ((location == (row * 7) - 1 and diff != 7) or (location == (row + 1) * 7 and diff != 7)
                            or location < 0 or location > 48 or location in body):
                        possible_action.append(action)
                return random.choice(possible_action) if len(possible_action) != 0 else random.randrange(self.action_size)

        output = self.model.predict(state)
        return np.argmax(output)

    def exp_replay(self):
        states = []
        target_fs = []
        next_states = []
        # current_states = []
        for event in self.memory:
            # current_states.append(event[0][0][0])
            for [_, _, _, next_state, done] in event:
                if not done:
                    next_states.append(next_state[0])
        next_outputs = deque(self.model.predict(np.array(next_states), verbose=1))  # .tolist()
        # state_outputs = deque(self.model.predict(np.array(current_states), verbose=1))  # .tolist()
        for event in self.memory:
            state = event[0][0]
            # target_f = state_outputs.popleft()
            target_f = [0] * 4
            for [_, action, reward, _, done] in event:
                if done:
                    target = reward
                else:
                    # target = reward + (self.gamma * max(next_outputs.popleft()))
                    target = max(min(reward + (self.gamma * max(next_outputs.popleft())), 1), -1)
                target_f[action] = target
            states.append(np.array(state[0][:]))
            target_fs.append(np.array(target_f[:]))
        # for [a, b] in zip(states, target_fs):
        #     print(f'     {int(b[0] * 1000)}')
        #     print(f'{int(b[1] * 1000)}         {int(b[3] * 1000)}')
        #     print(f'     {int(b[2] * 1000)}')
        #     for i in range(7):
        #         print(f'{round(a[(i * 7) + 0], 3)}    {round(a[(i * 7) + 1], 3)}    {round(a[(i * 7) + 2], 3)}    '
        #               f'{round(a[(i * 7) + 3], 3)}    {round(a[(i * 7) + 4], 3)}    {round(a[(i * 7) + 5], 3)}    '
        #               f'{round(a[(i * 7) + 6], 3)}    ')
        self.model.fit([states], [target_fs], epochs=1, verbose=1, batch_size=256)
