
ACTION = 0
GO = 1
STOP = -1
LEFT = 2
RIGHT = 3

total_number_of_games = 15
games_count = 0

x_train = np.array([])
y_train = np.array([])


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
        x_train = np.append(x_train, [values['old_closest_enemy']/1000])
        y_train = np.append(y_train, [values['action']])


    action = 0

    return action


  def gameover(self, score):
    global games_count
    games_count += 1

    if games_count >= total_number_of_games:
      # End game
      return

    # TODO: RESTART GAME


if __name__ == '__main__':
  w = Wrapper()


# TODO:
  # Create a function to "stop and restart" the game. Should just stop the vehicle, allow me to put it back on the trak and click run again.


