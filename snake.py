import numpy as np
import random
from math import floor


class snake:
    def __init__(self):
        self.start_location = 24
        self.location_history = []
        self.location_history.append(self.start_location)
        self.snake_length = 3
        self.food_location = 0
        self.generate_food()

    def play(self, move):
        if move == 0 or move == 'w':
            self.location_history.append(self.location_history[-1] - 7)
        elif move == 1 or move == 'a':
            self.location_history.append(self.location_history[-1] - 1)
        elif move == 2 or move == 's':
            self.location_history.append(self.location_history[-1] + 7)
        elif move == 3 or move == 'd':
            self.location_history.append(self.location_history[-1] + 1)
        self.location_history = self.location_history[-self.snake_length:]
        status = self.check_status()
        if 'invalid' in status:
            self.location_history.pop()
        return status

    def check_status(self):
        previous_loc = self.location_history[-2]
        current_loc = self.location_history[-1]
        diff = abs(current_loc - previous_loc)
        row = floor(previous_loc / 7)
        if (current_loc == (row * 7) - 1 and diff != 7) or (current_loc == (row + 1) * 7 and diff != 7)\
                or current_loc < 0 or current_loc > 48 or current_loc in self.location_history[:-1]:
            return 'invalid'
        elif current_loc == self.food_location:
            self.snake_length += 1
            self.generate_food()
            return 'food'
        else:
            return 'moved'

    def generate_food(self):
        empty = []
        for i in range(49):
            if i not in self.location_history:
                empty.append(i)
        self.food_location = random.choice(empty)

    def print_board(self):
        board = ['. '] * 49
        for e, i in enumerate(self.location_history):
            diff = abs(self.location_history[e] - self.location_history[e-1])
            board[i] = 'I ' if diff == 1 else 'H '
        board[self.location_history[-1]] = 'O '
        board[self.food_location] = 'X '
        for row in np.split(np.array(board), 7):
            board_row = ''
            for col in row:
                board_row += col
            print(board_row)
        print(' ')
