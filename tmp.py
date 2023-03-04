from pathlib import Path
import pandas as pd
import numpy as np
import os

def main():
    path = Path("A:/datasets/relation_extraction/benchmark/zs")

    print([f for f in os.listdir(path) if "dev" in f])
    df = pd.read_csv('A:/datasets/relation_extraction/benchmark/zs/dev.0', delimiter='\t', names=["relation", "masked_question", "entity", "context", "a1", "a2", "a3"], dtype=str)

    df["question"] = df.apply(lambda x: x.masked_question.replace("XXX", x.entity), axis=1)

    df = df.replace({np.nan: None})

    print(len(df.index))

if __name__ == "__main__":
    main()
