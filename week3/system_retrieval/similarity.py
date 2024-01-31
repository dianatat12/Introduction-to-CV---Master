from math import floor
from collections import Counter
from abc import abstractmethod
import numpy as np
import math


class SimilaritySearch:
    @abstractmethod
    def calculate(self, a, b):
        pass


class Distance(SimilaritySearch):
    # TODO: deal with similarity search vs distance
    def calculate(self, a, b):
        return -1 * self.calculate_distance(a, b)

    @abstractmethod
    def calculate_distance(self, a, b):
        pass

class L2DistanceTexture(Distance):
    def calculate_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.abs(a-b)**2)
    
    
class L1DistanceTexture(Distance):
    def calculate_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.abs(a-b))

class HistogramIntersection(SimilaritySearch):
    def calculate(self, a, b):
        return np.sum(np.minimum(a, b))


class Hellinger(Distance):
    def calculate_distance(self, a, b):
        return sum([(math.sqrt(t[0]) - math.sqrt(t[1])) * (math.sqrt(t[0]) - math.sqrt(t[1]))
                    for t in zip(a, b)]) / math.sqrt(2.)


class Chi2Distance(Distance):
    def calculate_distance(self, a, b):
        chi = 0
        for (a, b) in zip(a, b):
            if a + b != 0:
                chi += ((a - b) ** 2) / (a + b)

        return chi * 0.5


class L1Distance(Distance):
    def calculate_distance(self, a, b):
        return np.sum(np.abs(a - b))


class LevenshteinDistance(Distance):
    def calculate_distance(self, a, b):  # written by chat-gpt
        len_a = len(a)
        len_b = len(b)

        # Create a matrix to store the distances between substrings of a and b
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        # Initialize the first row and first column of the matrix
        for i in range(len_a + 1):
            dp[i][0] = i
        for j in range(len_b + 1):
            dp[0][j] = j

        # Fill in the matrix based on the recurrence relation
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution
                )

        # The Levenshtein distance is stored in the bottom-right cell of the matrix
        return dp[len_a][len_b]


class DamerauLevenshteinDistance(Distance):
    def calculate_distance(self, a, b):  # from https://www.geeksforgeeks.org/damerau-levenshtein-distance/
        # Create a table to store the results of subproblems
        dp = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

        # Initialize the table
        for i in range(len(a) + 1):
            dp[i][0] = i
        for j in range(len(b) + 1):
            dp[0][j] = j

        # Populate the table using dynamic programming
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        # Return the edit distance
        return dp[len(a)][len(b)]


class Jaro(Distance):
    def calculate(self, a, b):
        # If the s are equal
        if a == b:
            return 1.0

        # Length of two s
        len1 = len(a)
        len2 = len(b)

        # Maximum distance upto which matching
        # is allowed
        max_dist = floor(max(len1, len2) / 2) - 1

        # Count of matches
        match = 0

        # Hash for matches
        hash_a = [0] * len(a)
        hash_b = [0] * len(b)

        # Traverse through the first
        for i in range(len1):

            # Check if there is any matches
            for j in range(max(0, i - max_dist),
                           min(len2, i + max_dist + 1)):

                # If there is a match
                if a[i] == b[j] and hash_b[j] == 0:
                    hash_a[i] = 1
                    hash_b[j] = 1
                    match += 1
                    break

        # If there is no match
        if match == 0:
            return 0.0

        # Number of transpositions
        t = 0
        point = 0

        # Count number of occurrences
        # where two characters match but
        # there is a third matched character
        # in between the indices
        for i in range(len1):
            if hash_a[i]:

                # Find the next matched character
                # in second
                while hash_b[point] == 0:
                    point += 1

                if a[i] != b[point]:
                    t += 1
                point += 1
        t = t // 2

        # Return the Jaro Similarity
        return (match / len1 + match / len2 +
                (match - t) / match) / 3.0


class NwDistance(Distance):
    def calculate_distance(self, a, b, match=1, mismatch=1, gap=1):
        x, y = a, b
        nx = len(x)
        ny = len(y)
        # Optimal score at each possible pair of characters.
        F = np.zeros((nx + 1, ny + 1))
        F[:, 0] = np.linspace(0, -nx * gap, nx + 1)
        F[0, :] = np.linspace(0, -ny * gap, ny + 1)
        # Pointers to trace through an optimal aligment.
        P = np.zeros((nx + 1, ny + 1))
        P[:, 0] = 3
        P[0, :] = 4
        # Temporary scores.
        t = np.zeros(3)
        for i in range(nx):
            for j in range(ny):
                if x[i] == y[j]:
                    t[0] = F[i, j] + match
                else:
                    t[0] = F[i, j] - mismatch
                t[1] = F[i, j + 1] - gap
                t[2] = F[i + 1, j] - gap
                tmax = np.max(t)
                F[i + 1, j + 1] = tmax
                if t[0] == tmax:
                    P[i + 1, j + 1] += 2
                if t[1] == tmax:
                    P[i + 1, j + 1] += 3
                if t[2] == tmax:
                    P[i + 1, j + 1] += 4
        # Trace through an optimal alignment.
        i = nx
        j = ny
        rx = []
        ry = []
        while i > 0 or j > 0:
            if P[i, j] in [2, 5, 6, 9]:
                rx.append(x[i - 1])
                ry.append(y[j - 1])
                i -= 1
                j -= 1
            elif P[i, j] in [3, 5, 7, 9]:
                rx.append(x[i - 1])
                ry.append('-')
                i -= 1
            elif P[i, j] in [4, 6, 7, 9]:
                rx.append('-')
                ry.append(y[j - 1])
                j -= 1
        # Reverse the strings.
        rx = ''.join(rx)[::-1]
        ry = ''.join(ry)[::-1]
        return '\n'.join([rx, ry])


class GotohDistance(Distance):
    def calculate_distance(self, a, b, match_score=1, mismatch_score=-1, gap_open_penalty=-1,
                           gap_extension_penalty=-0.5):
        seq1, seq2 = a, b
        m, n = len(seq1), len(seq2)

        # Initialize the score matrix and traceback matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=float)
        traceback_matrix = np.zeros((m + 1, n + 1), dtype=int)

        # Initialize the first row and first column with gap penalties
        for i in range(1, m + 1):
            score_matrix[i][0] = gap_open_penalty + i * gap_extension_penalty
            traceback_matrix[i][0] = 1  # 1 indicates a gap
        for j in range(1, n + 1):
            score_matrix[0][j] = gap_open_penalty + j * gap_extension_penalty
            traceback_matrix[0][j] = 2  # 2 indicates a gap

        # Fill in the score and traceback matrices
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
                gap1 = score_matrix[i - 1][j] + (
                    gap_open_penalty if traceback_matrix[i - 1][j] != 1 else gap_extension_penalty)
                gap2 = score_matrix[i][j - 1] + (
                    gap_open_penalty if traceback_matrix[i][j - 1] != 2 else gap_extension_penalty)
                score_matrix[i][j] = max(match, gap1, gap2, 0)  # 0 is the minimum score

                if score_matrix[i][j] == match:
                    traceback_matrix[i][j] = 0  # 0 indicates a match or mismatch
                elif score_matrix[i][j] == gap1:
                    traceback_matrix[i][j] = 1  # 1 indicates a gap in the first sequence
                elif score_matrix[i][j] == gap2:
                    traceback_matrix[i][j] = 2  # 2 indicates a gap in the second sequence

        # Traceback to find the alignment and calculate the Gotoh distance
        alignment_seq1 = []
        alignment_seq2 = []
        i, j = m, n
        distance = 0

        while i > 0 or j > 0:
            if traceback_matrix[i][j] == 0:
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append(seq2[j - 1])
                if seq1[i - 1] != seq2[j - 1]:
                    distance += mismatch_score
                i, j = i - 1, j - 1
            elif traceback_matrix[i][j] == 1:
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append('-')
                distance += gap_open_penalty
                while traceback_matrix[i][j] == 1:
                    i -= 1
            elif traceback_matrix[i][j] == 2:
                alignment_seq1.append('-')
                alignment_seq2.append(seq2[j - 1])
                distance += gap_open_penalty
                while traceback_matrix[i][j] == 2:
                    j -= 1

        alignment_seq1 = ''.join(alignment_seq1[::-1])
        alignment_seq2 = ''.join(alignment_seq2[::-1])

        return distance, alignment_seq1, alignment_seq2


class SmithWatermanDistance(Distance):
    def calculate_distance(self, a, b, match_score=2, mismatch_score=-1, gap_penalty=-2):
        seq1, seq2 = a, b
        m, n = len(seq1), len(seq2)

        # Initialize the score matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=int)

        # Initialize variables to keep track of the maximum score and its position
        max_score = 0
        max_i, max_j = 0, 0

        # Fill in the score matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
                delete = score_matrix[i - 1][j] + gap_penalty
                insert = score_matrix[i][j - 1] + gap_penalty
                score_matrix[i][j] = max(0, match, delete, insert)

                if score_matrix[i][j] > max_score:
                    max_score = score_matrix[i][j]
                    max_i, max_j = i, j

        # Traceback to find the alignment and calculate the Smith-Waterman distance
        alignment_seq1 = []
        alignment_seq2 = []
        i, j = max_i, max_j
        distance = 0

        while i > 0 and j > 0 and score_matrix[i][j] > 0:
            current_score = score_matrix[i][j]
            match = score_matrix[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score)
            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty

            if current_score == match:
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append(seq2[j - 1])
                if seq1[i - 1] != seq2[j - 1]:
                    distance += mismatch_score
                i, j = i - 1, j - 1
            elif current_score == delete:
                alignment_seq1.append(seq1[i - 1])
                alignment_seq2.append('-')
                distance += gap_penalty
                i -= 1
            elif current_score == insert:
                alignment_seq1.append('-')
                alignment_seq2.append(seq2[j - 1])
                distance += gap_penalty
                j -= 1

        alignment_seq1 = ''.join(alignment_seq1[::-1])
        alignment_seq2 = ''.join(alignment_seq2[::-1])

        return distance, alignment_seq1, alignment_seq2


class MlipnsDistance(Distance):
    def calculate_distance(self, a, b):
        seq1, seq2 = a, b

        # Calculate the observed patterns in the sequences
        patterns1 = Counter(seq1)
        patterns2 = Counter(seq2)

        # Get the set of unique patterns from both sequences
        all_patterns = set(patterns1.keys()) | set(patterns2.keys())

        # Initialize variables to calculate the MLIPNS distance
        mlipns_numerator = 0
        mlipns_denominator1 = 0
        mlipns_denominator2 = 0

        for pattern in all_patterns:
            p1 = patterns1.get(pattern, 0)
            p2 = patterns2.get(pattern, 0)

            mlipns_numerator += min(p1, p2)
            mlipns_denominator1 += p1
            mlipns_denominator2 += p2

        if mlipns_denominator1 == 0 or mlipns_denominator2 == 0:
            return 0  # To handle division by zero

        mlipns_distance = 1 - (2 * mlipns_numerator) / (mlipns_denominator1 + mlipns_denominator2)

        return mlipns_distance


class HammingDistanceWithPadding(Distance):
    def calculate_distance(self, a, b, padding_char=' '):
        str1, str2 = a, b
        len1, len2 = len(str1), len(str2)
        max_len = max(len1, len2)

        # Pad the shorter string to the length of the longer string
        str1 = str1.ljust(max_len, padding_char)
        str2 = str2.ljust(max_len, padding_char)

        distance = 0
        for i in range(max_len):
            if str1[i] != str2[i]:
                distance += 1

        return distance
