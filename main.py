# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PPO_EMS import PPO_EMS
from environments.EMS_env import EMS_env
from json import load


def train():

    environment = EMS_env()
    rl_model = PPO_EMS(environment, options)
    rl_model.learn(200)


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
