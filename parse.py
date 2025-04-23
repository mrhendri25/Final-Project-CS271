import os
import pickle
from typing import Any, Callable
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


def getMainData() -> pd.DataFrame:
    if os.path.exists("./data/data.pkl"):
        with open("./data/data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        data = pd.read_csv("./data/cps_00011.csv")
        with open("./data/data.pkl", "wb") as f:
            pickle.dump(data, f)
    return data


def getPopScalers() -> pd.DataFrame:
    if os.path.exists("./data/per_cap.pkt"):
        with open("./data/per_cap.pkt", "rb") as f:
            sca = pickle.load(f)
    else:
        d = getMainData()
        d = d[pd.notna(d)]
        yrs, pcts = operate(
            d,
            "YEAR",
            "RACE",
            lambda x: {i: len([j for j in x if j == i]) / len(x) for i in set(x)},
        )
        sca = pd.DataFrame(pcts, index=yrs)
        with open("./data/per_cap.pkt", "wb") as f:
            pickle.dump(sca, f)
    return sca


def inflation(src: int, to: int) -> float:  # From FRED
    perYear: dict[int, float] = {
        2009: 1,
        2010: 1.640043442389900,
        2011: 3.156841568622000,
        2012: 2.069337265260670,
        2013: 1.464832655627170,
        2014: 1.622222977408170,
        2015: 0.118627135552451,
        2016: 1.261583205705360,
        2017: 2.130110003659610,
        2018: 2.442583296928170,
        2019: 1.812210075260210,
        2020: 1.233584396306290,
        2021: 4.697858863637420,
        2022: 8.002799820521210,
        2023: 4.116338383744880,
        2024: 2.949525204852070,
    }
    if src > to:
        y = to
        comp = src
    else:
        y = src
        comp = to
    acc = 1
    while y < comp:
        y += 1
        acc *= 1 + (perYear[y] / 100)
    if src > to:
        return 1 / acc
    else:
        return acc


def operate(
    df: pd.DataFrame,
    key: str | pd.Series | list[pd.Series],
    query: str,
    op: Callable[[pd.Series | pd.DataFrame | ndarray], Any] = lambda x: x,
) -> tuple[list, Any]:
    if type(key) is str:
        base = list(set(df[key]))
        base.sort()
        out = [op(df[df[key] == x][query]) for x in base]
    else:
        if type(key) is pd.Series:
            key = [key]
        assert type(key) is list[pd.Series]
        final = key.pop()
        while key != []:
            final = final & key.pop()
        out = op(df[final][query])
        base = []
    return base, out


mappings = {
    "race": {
        100: "White",
        200: "Black",
        300: "American Indian/Aleut/Eskimo",
        650: "Asian or Pacific Islander",
        651: "Asian only",
        652: "Hawaiian/Pacific Islander only",
        700: "Other (single race, n.e.c.",
        801: "White-Black",
        802: "White-American Indian",
        803: "White-Asian",
        804: "White-Hawaiian/Pacific Islander",
        805: "Black-American Indian",
        806: "Black-Asian",
        807: "Black-Hawaiian/Pacific Islander",
        808: "American Indian-Asian",
        809: "Asian-Hawaiian/Pacific Islander",
        810: "White-Black-American Indian",
        811: "White-Black-Asian",
        812: "White-American Indian-Asian",
        813: "White-Asian-Hawaiian/Pacific Islander",
        814: "White-Black-American Indian-Asian",
        815: "American Indian-Hawaiian/Pacific Islander",
        816: "White-Black--Hawaiian/Pacific Islander",
        817: "White-American Indian-Hawaiian/Pacific Islander",
    },
    "sex": {
        1: "male",
        2: "female",
    },
}

data = getMainData()
scalers = getPopScalers()

namask = pd.notna(data["INCWAGE"])
safe_data = data[namask]
big_mask = safe_data["INCWAGE"] <= 500000
white_ppl = safe_data["RACE"] == 100
black_ppl = safe_data["RACE"] == 200
z_mask = safe_data["INCWAGE"] != 0
print(scalers.index)
plt.plot(scalers.index, scalers)
plt.show()

# [print(x) for x in safe_data[safe_data["RACE"] == 100]["INCWAGE"]]
# plt.hist(
#     safe_data[black_ppl & big_mask & z_mask]["INCWAGE"],
#     bins=90,
#     color="red",
#     alpha=0.4,
# )
# plt.hist(
#     safe_data[white_ppl & big_mask & z_mask]["INCWAGE"],
#     bins=90,
#     color="blue",
#     alpha=0.4,
# )
# plt.show()
