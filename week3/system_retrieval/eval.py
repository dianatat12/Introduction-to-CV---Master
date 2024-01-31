from abc import abstractmethod
import ml_metrics
import numpy as np

class Evaluator:
    @abstractmethod
    def evaluate(self, gt, result, ds):
        pass


class MapkEvaluator(Evaluator):
    def __init__(self, k):
        self.k = k

    def evaluate(self, gt, result, ds):
        print(f"Evaluate at {self.k}")

        if 'qsd1' in ds:
            return ml_metrics.mapk(gt, result, self.k)
        elif 'qsd2' in ds:
            eval_result = []
            for a,p in zip(gt, result):
                for a_i, p_i in zip(a,p):
                    eval_result.append(ml_metrics.apk([a_i],p_i, self.k))
            return np.mean(eval_result)


