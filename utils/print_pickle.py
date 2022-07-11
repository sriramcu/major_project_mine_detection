import pickle
import pprint
import sys


def main():
    f = open(sys.argv[1], 'rb')
    
    while True:
        try:
            stored_obj = pickle.load(f)
        except EOFError:
            break

        pprint.pprint(stored_obj)

    f.close()

if __name__ == "__main__":
    main()