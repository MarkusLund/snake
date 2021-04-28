
from snake_env import Snake


if __name__ == '__main__':
    human = True
    env = Snake(human=human)

    if human:
        while True:
            env.run_game()
