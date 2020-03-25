import snake
from tensorflow.keras.models import load_model
import numpy as np
from time import sleep


def get_state(location_history, food_location):
    state = [([0] * 3) for i in range(49)]
    state[location_history[-1]][0] = 1  # location of head
    for location in location_history[0:-1]:
        state[location][1] = 1  # location of body
    state[food_location][2] = 1
    state = np.array(state).flatten()
    return np.array([state])


name = 'AI_model'
game = snake.snake()
ai = load_model(f'model/{name}')
end = False
while not end:
    current_history = game.location_history[:]
    current_food = game.food_location
    state = get_state(current_history, current_food)
    action = np.argmax(ai.predict(state))
    result = game.play(action)
    print('\n\n\n\n\n\n\n')
    game.print_board()
    if result == 'invalid':
        end = True
    sleep(.1)
