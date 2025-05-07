import os
import pickle
from typing import Any, Callable
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind


def reg(co: list, out):
    if any([len(x) == 0 for x in co]):
        print("Cannot do regression")
        return
    factors = np.array(co).T
    analysis = LinearRegression()

    analysis.fit(factors, out)

    print("coefficients \t", analysis.coef_)
    print("intercept \t", analysis.intercept_)
    print("R^2 value \t", analysis.score(factors, out))
    return analysis


def cat_to_dummy_reg(df, cat_vars=[], cont_vars=[]):
    df_encoded = df.copy()
    for var in cat_vars:
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=[var],
            prefix=var,
            drop_first=True,  # Drop whites / married / etc
        )

    predictors = [
        c
        for c in df_encoded.columns
        if any(c.startswith(var + "_") for var in cat_vars)
    ] + cont_vars
    co = [df_encoded[c].values for c in predictors]
    out = df_encoded["WAGE_2009"].values
    return co, out, predictors


def run_regression_set(
    df, cat_vars=[], cont_vars=[], stratify_by_age=False, by_year=False
):
    results = {}

    def do_reg(subdf, label):
        co, out, predictors = cat_to_dummy_reg(subdf, cat_vars, cont_vars)
        print(f"\nregression: {label}")
        model = reg(co, out)
        if model:
            results[label] = {
                "predictors": predictors,
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "r2": model.score(np.array(co).T, out),
            }

    label = f"Overall: {', '.join(cont_vars + cat_vars)}"
    do_reg(df, label)

    if stratify_by_age:
        for lo, hi in AGE_RANGES:
            subdf = df[df["AGE"].between(lo, hi)]
            label = f"Age {lo}-{hi}: {', '.join(cont_vars + cat_vars)}"
            do_reg(subdf, label)

    if by_year:
        for year in range(2009, 2025):
            subdf = df[df["YEAR"] == year]
            label = f"year {year}: {', '.join(cont_vars + cat_vars)}"
            do_reg(subdf, label)

    return results


def print_regression_summary(results, target_var=None, precision=3):
    for label, res in results.items():
        coefs = res["coefficients"]
        preds = res["predictors"]
        r2 = res["r2"]

        print(f"\n{label}")
        print(f"  R²: {round(r2, precision)}")

        if target_var:
            try:  # This try block is here because I keep making mistakes, kill me
                idx = preds.index(target_var)
                coef = round(coefs[idx], precision)
                print(f"  {target_var} coefficient: {coef}")
            except ValueError:
                print(f"  {target_var} not found in predictors")
        else:
            for pred, coef in zip(preds, coefs):
                print(f"  {pred}: {round(coef, precision)}")


def significanceTest(x, y):
    return ttest_ind(x, y, equal_var=False, alternative="two-sided")


def get_inputs(df, predictors):
    co = [df[var].values for var in predictors]
    out = df["WAGE_2009"].values
    return co, out


def getMainData() -> pd.DataFrame:
    if os.path.exists("./data/data.pkl"):
        with open("./data/data.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        data = pd.read_csv("./data/cps_00011.csv")
        with open("./data/data.pkl", "wb") as f:
            pickle.dump(data, f)
    return data


def getPopScalers(column: str) -> pd.DataFrame:
    file = f"./data/per_cap_{column}.pkt"
    if os.path.exists(file):
        with open(file, "rb") as f:
            sca = pickle.load(f)
    else:
        d = getMainData()
        d = d[pd.notna(d)]
        yrs, pcts = operate(
            d,
            "YEAR",
            column,
            lambda x: {i: len([j for j in x if j == i]) / len(x) for i in set(x)},
        )
        sca = pd.DataFrame(pcts, index=yrs)
        with open(file, "wb") as f:
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
    df: pd.DataFrame | pd.Series,
    key: str | pd.Series | list[pd.Series],
    query: str,
    op: Callable[[pd.Series | pd.DataFrame | ndarray, Any], Any] = lambda x, y: x,
) -> tuple[list, Any]:
    if type(key) is str:
        base = list(set(df[key]))
        base.sort()
        out = [op(df[df[key] == x][query], x) for x in base]
    else:
        if type(key) is pd.Series:
            key = [key]
        assert type(key) is list[pd.Series]
        final = key.pop()
        while key != []:
            final = final & key.pop()
        out = op(df[final][query], final)
        base = []
    return base, out


mappings: dict[str, dict[int, str]] = {
    "race": {
        100: "White",
        200: "Black",
        300: "American Indian/Aleut/Eskimo",
        651: "Asian only",
        652: "Hawaiian/Pacific Islander only",
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
    "marital_status": {
        1: "Married, spouse present",
        2: "Married, spouse absent",
        3: "Separated",
        4: "Divorced",
        5: "Widowed",
        6: "Never married/single",
        9: "NIU",
    },
}


data = getMainData()
race_scalers = getPopScalers("RACE")
sex_scalers = getPopScalers("SEX")
age_scalers = getPopScalers("AGE")
marital_scalers = getPopScalers("MARST")
namask = pd.notna(data["INCWAGE"])
safe_data = data[namask]
big_mask = safe_data["INCWAGE"] <= 500000
z_mask = safe_data["INCWAGE"] != 0
filt = safe_data[big_mask & z_mask]


# inc_by_race_year_adj = {
#     r: operate(
#         filt[safe_data["RACE"] == r],
#         "YEAR",
#         "INCWAGE",
#         lambda x, y: np.median(x) * inflation(y, 2009)
#         if race_scalers[r][y] > 0.001 and len(x) > 200
#         else 0,
#     )
#     for r in racemap
# }

# for yr in range(2009, 2025):
#     print(f"Year: {yr}\n------")
#     year_control = filt["YEAR"] == yr
#     for race in [r for r in mappings["race"] if race_scalers[r][2009] > 0.005]:
#         print(f"\nRace: {mappings['race'][race]}\n----")
#         race_control = filt["RACE"] == race
#         for marstat in [
#             m for m in mappings["marital_status"] if marital_scalers[m][2009] > 0.005
#         ]:
#             print(f"\nMarital Status: {mappings['marital_status'][marstat]}--")
#             mar_control = filt["MARST"] == marstat
#             age_cut = filt["AGE"] < 65
#             iter_data = filt[year_control & race_control & mar_control & age_cut]
#             inc = iter_data["INCWAGE"]
#             age = iter_data["AGE"]
#             sex = iter_data["SEX"]
#             reg([sex, age], inc)


YEARS = range(2009, 2025)
AGE_RANGES = [(18, 25), (26, 35), (36, 45), (46, 55), (56, 65)]

# precompute inflation-adjusted wages
filt["WAGE_2009"] = filt.apply(
    lambda row: row["INCWAGE"] * inflation(row["YEAR"], 2009), axis=1
)
# My data probing, left for posterity
#
#
#
#
#
# significant_years = {y: [] for y in range(2009, 2025)}
# for y1 in YEARS:
#     for y2 in range(y1 + 1, 2025):
#         w1 = filt[filt["YEAR"] == y1]["WAGE_2009"]
#         w2 = filt[filt["YEAR"] == y2]["WAGE_2009"]
#         t, p = significanceTest(w1, w2)
#         if float(format(p)) < 0.05:
#             # print(f"{y1} vs {y2}: SIGNIFICANT (p={p:.3g}, t={t:.3g})")
#             significant_years[y1].append(y2)
#             # significant_years[y2].append(y1)
#
# for y1, comps in significant_years.items():
#     for y2 in comps:
#         for lo, hi in AGE_RANGES:
#             a1 = filt[(filt["YEAR"] == y1) & (filt["AGE"].between(lo, hi))]["WAGE_2009"]
#             a2 = filt[(filt["YEAR"] == y2) & (filt["AGE"].between(lo, hi))]["WAGE_2009"]
#             t, p = significanceTest(a1, a2)
#             if p < 0.05:
#                 print(f"{y1} vs {y2} sig controlling for {lo}-{hi} ages (p={p:.3g})")
#
# # plot avg wage per age group over time
# # for lo, hi in AGE_RANGES:
# #     group_means = [
# #         filt[(filt["YEAR"] == y) & filt["AGE"].between(lo, hi)]["WAGE_2009"].mean()
# #         for y in YEARS
# #     ]
# #     plt.plot(YEARS, group_means, label=f"{lo}-{hi}")
# # plt.legend()
# # plt.show()
#
# # cross-race t-tests (2009 only)
# races = list(mappings["race"].keys())
# for i, r1 in enumerate(races):
#     for r2 in races[i + 1 :]:
#         if race_scalers[r2][2009] < 0.001:
#             continue
#         w1 = filt[(filt["RACE"] == r1) & (filt["YEAR"] == 2009)]["INCWAGE"]
#         w2 = filt[(filt["RACE"] == r2) & (filt["YEAR"] == 2009)]["INCWAGE"]
#         t, p = ttest_ind(w1, w2, equal_var=False)
#         sig = "SIGNIFICANT" if float(format(p)) < 0.05 else "Not significant"
#         print(
#             f"{mappings['race'][r1]} vs {mappings['race'][r2]}: {sig} (p={p:.3g}, t={t:.3g})"
#         )
#
# for y in YEARS:
#     for lo, hi in AGE_RANGES:
#         for sex1 in [1]:
#             sex2 = 2
#             group1 = filt[
#                 (filt["YEAR"] == y)
#                 & (filt["SEX"] == sex1)
#                 & filt["AGE"].between(lo, hi)
#             ]["WAGE_2009"]
#             group2 = filt[
#                 (filt["YEAR"] == y)
#                 & (filt["SEX"] == sex2)
#                 & filt["AGE"].between(lo, hi)
#             ]["WAGE_2009"]
#             if len(group1) > 30 and len(group2) > 30:  # sanity check
#                 t, p = ttest_ind(group1, group2, equal_var=False)
#                 if p < 0.05:
#                     print(
#                         f"{y}: {mappings['sex'][sex1]} vs {mappings['sex'][sex2]} in age {lo}-{hi} is SIGNIFICANT (p={p:.3g}, t={t:.3g})"
#                     )
#
# marry_vals = [k for k in mappings["marital_status"] if k != 9]
# for y in YEARS:
#     for lo, hi in AGE_RANGES:
#         for i, m1 in enumerate(marry_vals):
#             for m2 in marry_vals[i + 1 :]:
#                 group1 = filt[
#                     (filt["YEAR"] == y)
#                     & (filt["MARST"] == m1)
#                     & filt["AGE"].between(lo, hi)
#                 ]["WAGE_2009"]
#                 group2 = filt[
#                     (filt["YEAR"] == y)
#                     & (filt["MARST"] == m2)
#                     & filt["AGE"].between(lo, hi)
#                 ]["WAGE_2009"]
#                 if len(group1) > 30 and len(group2) > 30:
#                     t, p = ttest_ind(group1, group2, equal_var=False)
#                     if p < 0.05:
#                         print(
#                             f"{y}: {mappings['marital_status'][m1]} vs {mappings['marital_status'][m2]} in age {lo}-{hi} is SIGNIFICANT (p={p:.3g}, t={t:.3g})"
#                         )


race_results = run_regression_set(filt, cat_vars=["RACE"])
race_age_year_results = run_regression_set(
    filt, cat_vars=["RACE"], cont_vars=["AGE"], by_year=True
)
race_age_results = run_regression_set(
    filt,
    cat_vars=["RACE"],
    cont_vars=["AGE"],
)

sex_results = run_regression_set(filt, cont_vars=["SEX"])
age_overall = run_regression_set(filt, cont_vars=["AGE"])
marst_age_results = run_regression_set(
    filt, cat_vars=["MARST"], cont_vars=["AGE"], stratify_by_age=True
)
full_model_results = run_regression_set(
    filt,
    cat_vars=["RACE", "MARST"],
    cont_vars=["AGE", "SEX"],
    stratify_by_age=True,
    by_year=True,
)

print_regression_summary(race_results)
print_regression_summary(sex_results)
print_regression_summary(age_overall)
print_regression_summary(race_age_results)
print_regression_summary(race_age_year_results)
print_regression_summary(marst_age_results)
print_regression_summary(full_model_results)
