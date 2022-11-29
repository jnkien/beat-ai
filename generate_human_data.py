""" Let a human play the game and record actions and states """

import lib.env
import lib.model
import lib.play_human
import lib.utils


def main():
    """ Main function
    """
    env = lib.env.create_unstacked_env()
    save_callback = lib.play_human.SaveCallback("./data/human/run1")
    lib.play_human.play_human(env, save_callback.call)

if __name__ == '__main__':
    main()
