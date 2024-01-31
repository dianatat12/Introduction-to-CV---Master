from abc import abstractmethod
import ml_metrics
import numpy as np
import re


class Evaluator:
    @abstractmethod
    def evaluate(self, gt, result):
        pass


class MapkEvaluator(Evaluator):
    def __init__(self, k):
        self.k = k

    def extract_number_from_string(self,input_string):
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

    def evaluate(self, gt, result):
        print(f"Evaluate at {self.k}")
        
        final_result = []
        for line in result:
            partial = []
            for part in line:
                best_k = dict(sorted(part.items(), key=lambda item: item[1], reverse=True)[:10])
                res = all(x == -1 for x in best_k.values())
                if res:
                    ll =[-1] * 10
                else:
                    ll = [self.extract_number_from_string(key) for key in best_k.keys()]
                partial.append(ll)
            final_result.append(partial)
            
        print("resultat final pre eval", final_result)
        
        result_final = []
        for a, p in zip(gt, final_result):
            for a_i, p_i in zip(a, p):
                result_final.append(ml_metrics.apk([a_i], p_i, self.k))
                
        end = np.mean(result_final)
        # return ml_metrics.mapk(gt, result, self.k)
        return end
