# Snake AI
Gameplay
-----
![](gif/play.gif)

Purpose
------
Create an AI using Deep Q-learning to play snake.

Input / Output
------
The input contains 147 features since there are 49 locations, and each location contains 3 features. (head, body, and food)\
The output is the 4 possible moves that it can make (up, down, left, or right).

Result
------
After training, it learns to find the food and avoid its own body most of the time. 
However, it can not plans very far ahead.