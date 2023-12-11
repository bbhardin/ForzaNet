import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
from random import randint

from game import controlled_run

ACTION = 0 # Do nothing
GO = 4
STOP = -1
LEFT = 2
RIGHT = 3
JUMP = 1
DOUBLE_JUMP = 4

PERFECT_SCORE = 1000

total_number_of_games = 2000
games_count = 0

x_train = np.array([])
y_train = np.array([])

train_frequency = 120   # If we assume 60 fps (which is unlikely on Xbox cloud, this is every 2 seconds)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# So I think right now I'm predicting jump or don't jump; 1 or 0. But really I want to be predicting 
# left, right, gas, brake. So I'll need more categories and possible options

# To test a simple version of this, I'll start by creating a double_jump function


class Wrapper(object):
  
  def __init__(self):
    print("Started the game")
    # TODO: Start the game
    controlled_run(self, 0)
    #controlled_run(self, 0)

  def control(self, values):
    global x_train
    global y_train

    # Func that is called by the game
    print(values)

    if values['closest_enemy'] == -1:
      return ACTION

    if values['old_closest_enemy'] is not -1:
      if values ['score_increased'] == 1:
        x_train = np.append(x_train, [values['old_closest_enemy']/PERFECT_SCORE])
        y_train = np.append(y_train, [values['action']])

    # The prediction from neural network
    prediction2 = model.predict(np.array([values['closest_enemy']/PERFECT_SCORE]))
    #print(prediction2)
    prediction = np.argmax(prediction2, axis=1)#model.predict_classes(np.array([[values['closest_enemy']]])/PERFECT_SCORE)
    #print(prediction)

    r = randint(0, 100)

    # Add randomness in early stages of training
    random_rate = 50*(1-games_count/50)

    if r < random_rate - 20:
      if prediction == ACTION:
        return DOUBLE_JUMP
      else:
        return ACTION
    elif r < random_rate and r > random_rate - 20:
      return DOUBLE_JUMP
    else:
      if prediction == JUMP:
        return JUMP
      else:
        return ACTION

    # Should never be reached
    return prediction


  def gameover(self, score):
    global games_count
    global x_train
    global y_train
    global model

    games_count += 1

    if games_count is not 0 and games_count % train_frequency is 0:
        # Before training, make the y_train array categorical
        y_train_cat = to_categorical(y_train, num_classes = 5)

        # Train the net
        model.fit(x_train, y_train_cat, epochs=50, verbose=1, shuffle=1)

        x_train = np.array([])
        y_train = np.array([])


    if games_count >= total_number_of_games:
      # End game
      return

    # TODO: RESTART GAME
    print("Starting another game")
    controlled_run(self, games_count)


if __name__ == '__main__':
  w = Wrapper()


# TODO:
  # Create a function to "stop and restart" the game. Should just stop the vehicle, allow me to put it back on the trak and click run again.


