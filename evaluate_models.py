
from pickle import load
import sys
from pathlib import Path
import os
import pandas as pd

def main(args):

    # Make the data from this path
    path = Path(args[0])

    for model in os.listdir("./models"):
        print(model)
        with open(os.path.join("./models", model), "rb") as f:
            model = load(f)

        # Score model on data loaded from path

if __name__ == '__main__':
    
    main(sys.argv[1:]) 