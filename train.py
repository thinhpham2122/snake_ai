import snake
import agent
import numpy as np


def get_state(location_history, food_location):
    state = [0] * 49
    length = len(location_history)
    for n, i in enumerate(location_history):
        state[i] = ((length / 100) - 1) - ((n + 1) / 100)
    state[food_location] = 1
    return np.array([state])


def get_event(location_history, food_location):
    possible_actions = 4
    events = []
    state = get_state(location_history, food_location)
    game_e = snake.snake()
    for action in range(possible_actions):
        game_e.location_history = location_history[:]
        game_e.food_location = food_location
        game_e.snake_length = max(len(location_history), 3)
        result = game_e.play(action)
        reward = get_reward(result)
        if reward == -1:
            done = True
            events.append([None, action, reward, None, done])
        else:
            next_state = get_state(game_e.location_history, game_e.food_location)
            done = False
            events.append([None, action, reward, next_state[:], done])

    events[0][0] = state
    return events


def get_reward(result):
    if 'move' in result:
        return 0
    if 'food' in result:
        return 1
    if 'invalid' in result:
        return -1


def test(model):
    game_test = snake.snake()
    end = False
    move = 0
    while not end:
        state = get_state(game_test.location_history, game_test.food_location)
        prediction = model.predict(state)[0]
        action = np.argmax(prediction)
        result = game_test.play(action)
        print(f'     {int(prediction[0]*1000)}')
        print(f'{int(prediction[1]*1000)}     {int(prediction[3]*1000)}')
        print(f'     {int(prediction[2]*1000)}')
        game_test.print_board()
        move += 1
        if result == 'invalid' or move > 1000:
            end = True


def train():
    ai = agent.Agent(49, 4, model_name=name)
    game_n = 0
    while True:
        game = snake.snake()
        end = False
        game_n += 1
        move = 0
        print(game_n, len(ai.memory), end='\r')
        while not end:
            current_history = game.location_history[:]
            current_food = game.food_location
            state = get_state(current_history, current_food)
            action = ai.act(state)
            result = game.play(action)
            ai.memory.append(get_event(current_history, current_food))
            move += 1
            if result == 'invalid' or move > 1000:
                end = True
        if game_n % 10000 == 0:
            ai.exp_replay()
            test(ai.model)
            ai.model.save(f'model/{name}_{game_n}')


name = 'b'
train()
