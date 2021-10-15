"""
Conditional Probability Activity & Exercise
"""
from numpy.random import seed, choice, random


def wrapper():
    """
    wrapper function
    """
    seed(0)
    totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
    purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
    total_purchases = 0
    for _ in range(100000):
        age_decade = choice([20, 30, 40, 50, 60, 70])
        purchase_probability = float(age_decade) / 100.0
        totals[age_decade] += 1
        if random() < purchase_probability:
            total_purchases += 1
            purchases[age_decade] += 1
    print(totals)
    print(purchases)
    print(total_purchases)
    pef = float(purchases[30]) / float(totals[30])
    print("P(purchase | 30s): " + str(pef))
    prof = float(totals[30]) / 100000.0
    print("P(30's): " + str(prof))
    proe = float(total_purchases) / 100000.0
    print("P(Purchase):" + str(proe))
    print("P(30's, Purchase)" + str(float(purchases[30]) / 100000.0))
    print("P(30's)P(Purchase)" + str(proe * prof))
    print((purchases[30] / 100000.0) / prof)

    totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
    purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
    total_purchases = 0
    for _ in range(100000):
        age_decade = choice([20, 30, 40, 50, 60, 70])
        purchase_probability = 0.4
        totals[age_decade] += 1
        if random() < purchase_probability:
            total_purchases += 1
            purchases[age_decade] += 1
    pef = float(purchases[30]) / float(totals[30])
    print("P(purchase | 30s): " + str(pef))
    proe = float(total_purchases) / 100000.0
    print("P(Purchase):" + str(proe))


if __name__ == "__main__":
    wrapper()
