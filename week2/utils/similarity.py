import numpy as np
import math
import re


def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))


def chi2_distance(a, b):
    chi = 0
    for (a, b) in zip(a, b):
        if a + b != 0:
            chi += ((a - b) ** 2) / (a + b)

    return chi * 0.5


def l1_distance(hist1, hist2):
    return np.sum(np.abs(hist1 - hist2))


def hellinger(p, q):
    return sum([(math.sqrt(t[0]) - math.sqrt(t[1])) * (math.sqrt(t[0]) - math.sqrt(t[1]))
                for t in zip(p, q)]) / math.sqrt(2.)



# HELPERS
def intersection_per_query(hist_query, hist_bbdd: dict, k):
    result_query = {}
    for key, value in hist_bbdd.items():
        result_query[key] = histogram_intersection(hist_query, value)

    best_k = dict(sorted(result_query.items(), key=lambda item: item[1], reverse=True)[:k])
    return [extract_number_from_string(key) for key in best_k.keys()]


# Written by chatGPT
def extract_number_from_string(input_string):
    # Define a regular expression pattern to match the number part.
    pattern = r'\d+'

    # Use re.search to find the first match in the input string.
    match = re.search(pattern, input_string)

    # Check if a match was found.
    if match:
        # Extract the matched number as a string.
        number_str = match.group(0)

        # Convert the extracted string to an integer.
        number = int(number_str)

        return number
    else:
        # Return None if no match was found.
        return None

