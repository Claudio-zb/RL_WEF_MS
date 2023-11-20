# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PPO import PPO
from environments.EMS_env import EMS_env
from environments.Quad_env import Quad_env
from json import load


def train():

    environment = Quad_env()
    rl_model = PPO(environment, options)
    rl_model.learn(400)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        with open('./train_options.json', 'r') as file:
            options = load(file)
        print(options)
        print("Options loaded")
    except Exception as e:
        print(e)
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
