import lib.env
import lib.model
import lib.utils
import lib.play_human

def main():
    env = lib.env.create_unstacked_env()    
    save_callback = lib.play_human.SaveCallback("./data/human/run1")
    lib.play_human.play_human(env, save_callback.call)

if __name__ == '__main__':
    main()