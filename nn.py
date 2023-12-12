import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
from random import randint
import matplotlib as plt

from game import controlled_run

ACTION = 0 # Do nothing
GO = 4
STOP = -1
LEFT = 2
RIGHT = 3
JUMP = 1
DOUBLE_JUMP = 4

PERFECT_SCORE = 1000

total_number_of_games = 200
games_count = 0

x_train = np.array([])
y_train = np.array([])

train_frequency = 10   # If we assume 60 fps (which is unlikely on Xbox cloud, this is every 2 seconds)

# Simple linear classification with 2 layers
model = Sequential()
#input_tensor = keras.layers.Normalization(input_shape=[5,], axis=None) # Define the input shape
model.add(Dense(1, input_dim=1, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))
model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# So I think right now I'm predicting jump or don't jump; 1 or 0. But really I want to be predicting 
# left, right, gas, brake. So I'll need more categories and possible options

# To test a simple version of this, I'll start by creating a double_jump function


# matplot functions
fig, _ = plt.subplot(ncols=1, nrows=3, figsize=(6, 6))
fig.tight_layout()

all_scores = []
avg_scores = []
avg_score_rate = []
all_x, all_y = np.array([]), np.array([])


class Wrapper(object):
  
  def __init__(self):
    print("Started the game")
    controlled_run(self, 0)

  @staticmethod
  def visualize():
    global all_x
    global all_y
    global avg_scores
    global all_scores
    global x_train
    global y_train

    plt.subplot(3, 1, 1)
    x = np.linspace(1, len(all_scores), len(all_scores))
    plt.plot(x, all_scores, 'o-', color = 'g')
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.title("Score per game")

    plt.subplot(3, 1, 2)
    plt.scatter(x_train[y_train==0], y_train[y_train==0], color='r', label='Stay still')
    plt.scatter(x_train[y_train==1], y_train[y_train==0], color='b', label='Jump')
    plt.scatter(x_train[y_train==4], y_train[y_train==0], color='g', label='Double jump')
    plt.xlabel('Distance from the nearest enemy')
    plt.title('Training data')

    plt.subplot(3, 1, 3)
    x2 = np.linspace(1, len(average_scores), len(average_scores))
    plt.plot(x2, average_scores, 'o-', color='b')
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.title("Average scores per 10 games")

    plt.pause(0.001)


  def control(self, values):
    global x_train
    global y_train

    # Func that is called by the game
    print('values:', values)

    if values['closest_enemy'] == -1:
      return ACTION

    if values['old_closest_enemy'] is not -1:
      if values ['score_increased'] == 1:
        x_train = np.append(x_train, [values['old_closest_enemy']/PERFECT_SCORE])
        y_train = np.append(y_train, [values['action']])

    # The prediction from neural network
    #print('array: ', np.array([values['closest_enemy']/PERFECT_SCORE, 0, 0, 0, 0]))
    prediction2 = model.predict(np.array([values['closest_enemy']/PERFECT_SCORE]))
    prediction = np.argmax(prediction2, axis=1)#model.predict_classes(np.array([[values['closest_enemy']]])/PERFECT_SCORE)
    #print(prediction)


    # Add randomness in early stages of training
    r = randint(0, 100)
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

    global all_x
    global all_y
    global all_scores
    global average_scores
    global average_score_rate

    games_count += 1

    all_x = np.append(all_x, x_train)
    all_y = np.append(all_y, y_train)

    all_scores.append(score)

    Wrapper.visualize()

    if games_count is not 0 and games_count % average_score_rate is not 0:
      average_score = sum(all_scores) / len(all_scores)
      average_scores.append(average_score)

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


