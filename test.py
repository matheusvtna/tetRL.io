import argparse
import torch
import time
from src.tetris import Tetris
import os

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--checkpoint_file", type=str, default="tetris")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    CHECKPOINT_FILE = opt.saved_path + "/" + opt.checkpoint_file

    if os.path.isfile(CHECKPOINT_FILE):
        print("--> Carregando Checkpoint '{}'.".format(CHECKPOINT_FILE))
    else:
        print("--> Checkpoint '{}' não encontrado.".format(CHECKPOINT_FILE))
        return

    if torch.cuda.is_available():
        model = torch.load("{}/{}".format(opt.saved_path, opt.checkpoint_file))
    else:
        model = torch.load("{}/{}".format(opt.saved_path, opt.checkpoint_file), map_location=lambda storage, loc: storage)
    
    model.eval()
    env = Tetris(width=opt.width, height=opt.height)
    env.reset()
    
    if torch.cuda.is_available():
        model.cuda()

    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True)

        time.sleep(0.5)

        if done:
            env.reset()
        


if __name__ == "__main__":
    opt = get_args()
    test(opt)
