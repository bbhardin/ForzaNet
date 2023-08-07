import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np

ACTION = 0 # Do nothing
GO = 1
STOP = -1
LEFT = 2
RIGHT = 3

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


class Wrapper(object):
  
  def __init__(self):
    # TODO: Start the game
    # controlled_run(self, 0)


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
    prediction = model.predict_classes(np.array([[values['closest_enemy']]])/PERFECT_SCORE)

    r = randint(0, 100)

    # Add randomness in early stages of training
    random_rate = 100*(1-games_count/100)

    if r < random_rate:
      if prediction == ACTION:
        return GO
      else:
        return ACTION
    else:
      if prediction == JUMP:
        return JUMP
      else:
        return ACTION

    return prediction


  def gameover(self, score):
    global games_count
    global x_train
    global y_train
    global model

    games_count += 1

    if games_count is not 0 and games_count % train_frequency is 0:
        # Before training, make the y_train array categorical
        y_train_cat = to_categorical(y_train, num_clASSES = 2)

        # Train the net
        model.fit(x_train, y_train_cat, epochs=50, verbose=1, shuffle=1)

        x_train = np.array([])
        y_train = np.array([])


    if games_count >= total_number_of_games:
      # End game
      return

    # TODO: RESTART GAME


if __name__ == '__main__':
  w = Wrapper()


# TODO:
  # Create a function to "stop and restart" the game. Should just stop the vehicle, allow me to put it back on the trak and click run again.


