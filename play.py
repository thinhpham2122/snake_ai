import snake
from tensorflow.keras.models import load_model
import numpy as np
from time import sleep


def get_state(location_history, food_location):
    state = [0] * 49
    length = len(location_history)
    for n, i in enumerate(location_history):
        state[i] = ((length / 100) - 1) - ((n + 1) / 100)
    state[food_location] = 1
    return np.array([state])


name = 'b_140000'
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
    sleep(.3)
