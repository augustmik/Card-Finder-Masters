import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

SIZE_OF_MAP = 9

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
META = 4

L_REWARD = 100
T_REWARD = 1
N_REWARD = -10
STEP_REWARD = -1

terminal_output = open(1, 'w')

def generate_random_map(size=SIZE_OF_MAP, p=0.3, tp=0.05):
    """Generuoja atsitiktinį tinkamą žemėlapį
    :param size: kiekvienos žemėlapio kraštinės dydis
    :param p: tikimybė, kad kortele neutrali
    """
    valid = False
    while not valid:
        p = min(1, p)
        res = np.random.choice(["-", "T", "N"], (size, size), p=[p, tp, 1-tp-p])
        # res = np.random.choice(["-", "N"], (size, size), p=[p, 1-p])
        jumpers = np.random.choice(size*size, (size, size), replace=False)
        res[0][0] = "S"
        res[-1][-1] = "L"
        valid = True
    return ["".join(x) for x in res], jumpers


class CardFinderEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, desc=None):
        desc, jumpers = generate_random_map()
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (N_REWARD, L_REWARD)
        print(jumpers)
        nA = 5
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        
        def to_s(row, col):
            return row * ncol + col

        def jumpWithMeta(row, col):
            jumpToCard = jumpers[row,col]
            newRow = jumpToCard//SIZE_OF_MAP
            newCol = jumpToCard-(newRow*SIZE_OF_MAP)
            return newRow, newCol

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            elif a == META:
                row, col = jumpWithMeta(row, col)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"L"
            if newletter == b"L":
                reward = L_REWARD
            elif newletter == b"N":
                reward = N_REWARD
            elif newletter == b"T":
                reward = T_REWARD
            else: 
                reward = STEP_REWARD
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"L":
                        li.append((1, s, L_REWARD, True))
                    else:
                        li.append((1, *update_probability_matrix(row, col, a)))

        super(CardFinderEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up", "MetaData"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
