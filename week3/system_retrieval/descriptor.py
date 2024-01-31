from tqdm import tqdm
from abc import abstractmethod
import utils
import cv2
import uuid
import os
import pickle

class Descriptor:
    def __init__(self, ss_calculator, desc_function, dsno):
        self.ss_calculator = ss_calculator
        self.desc_function = desc_function
        self.dsno = dsno

    @abstractmethod
    def calculate_queries(self, queries):
        pass

    @abstractmethod
    def calculate_bbdd(self, bbdd):
        pass

    def calculate_ss_per_query(self, desc_queries, desc_bbdd, k=10):
        result_query = {}

        for bbdd_name, bbdd_descriptor in desc_bbdd.items():
            ss = self.ss_calculator.calculate(desc_queries, bbdd_descriptor)
            result_query[bbdd_name] = ss

        return result_query

    def calculate_ss(self, desc_queries, desc_bbdd):
        result = []

        progress_bar = tqdm(total=len(desc_queries), unit="iteration", position=0)
        for query_name, query_descriptor in desc_queries.items():

            for p in query_descriptor:
                ss = self.calculate_ss_per_query(p, desc_bbdd)
                result.append(ss)

            progress_bar.update(1)

        return result
    
    def calculate(self, queries, bbdd):
        print("Calculate query descriptors")
        desc_queries = self.calculate_queries(queries)

        print("Calculate BBDD descriptors")
        desc_bbdd = self.calculate_bbdd(bbdd)

        print("Measure similarity")
        ss_result = self.calculate_ss(desc_queries, desc_bbdd)

        return ss_result



class TextDescriptor(Descriptor):
    def __init__(self, ss_calculator, desc_function, dsno):
        super().__init__(ss_calculator, desc_function, dsno)

    def calculate_queries(self, queries):
        descriptor_queries = {}

        progress_bar = tqdm(total=len(queries), unit="iteration", position=0)
        for query_name, query_img in queries.items():
            
            img, _, bb_text = query_img

            descriptor_queries[query_name] = []
            for i, t in zip(img, bb_text):

                desc = self.desc_function(i, t)
                descriptor_queries[query_name].append(desc)

            progress_bar.update(1)

        return descriptor_queries

    def calculate_bbdd(self, bbdd):
        return bbdd


class HistogramDescriptor(Descriptor):
    def __init__(self, ss_calculator, desc_function, dsno):
        super().__init__(ss_calculator, desc_function, dsno)

    def calculate_queries(self, queries):
        desc_queries = {}

        prep_query_path = 'results/' + self.dsno + '_queries_hist'

        if os.path.isfile(prep_query_path):
            with open(prep_query_path, 'rb') as file:
                desc_queries = pickle.load(file)
        else:
            progress_bar = tqdm(total=len(queries), unit="iteration", position=0)
            for query_name, query_img in queries.items():
                img, _, bb_text = query_img

                desc_queries[query_name] = []
                for i, t in zip(img, bb_text):
                    desc = self.desc_function(i, with_bbox=True, text_box=t)
                    desc_queries[query_name].append(desc)

                progress_bar.update(1)

            with open(prep_query_path, 'wb') as file:
                pickle.dump(desc_queries, file)

        return desc_queries

    def calculate_bbdd(self, bbdd):
        desc_bbdd = {}

        prep_result_path = 'results/' + self.dsno + '_bbdd_hist'
        if os.path.isfile(prep_result_path):
            with open(prep_result_path, 'rb') as file:
                desc_bbdd = pickle.load(file)
        else:
            progress_bar = tqdm(total=len(bbdd), unit="iteration", position=0)
            for bbdd_name, img in bbdd.items():
                desc = self.desc_function(img)
                desc_bbdd[bbdd_name] = desc

                progress_bar.update(1)

            with open(prep_result_path, 'wb') as file:
                pickle.dump(desc_bbdd, file)

        return desc_bbdd


class TextureDescriptor(Descriptor):
    def __init__(self, ss_calculator, desc_function, dsno):
        super().__init__(ss_calculator, desc_function, dsno)

    def calculate_queries(self, queries):
        desc_queries = {}

        prep_query_path = 'results/' + self.dsno + '_queries_texture'

        if os.path.isfile(prep_query_path):
            with open(prep_query_path, 'rb') as file:
                desc_queries = pickle.load(file)

        else:
            progress_bar = tqdm(total=len(queries), unit="iteration", position=0)
            for query_name, query_img in queries.items():
                img, _, _ = query_img

                desc_queries[query_name] = []
                for i in img:

                    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                    desc = self.desc_function(i)
                    desc_queries[query_name].append(desc)

                progress_bar.update(1)

            with open(prep_query_path, 'wb') as file:
                pickle.dump(desc_queries, file)

        return desc_queries

    def calculate_bbdd(self, bbdd):
        desc_bbdd = {}

        prep_result_path = 'results/' + self.dsno + '_bbdd_texture'
        if os.path.isfile(prep_result_path):
            with open(prep_result_path, 'rb') as file:
                desc_bbdd = pickle.load(file)
        else:
            bbdd = utils.convert_color_space(bbdd, cv2.COLOR_BGR2GRAY)
            progress_bar = tqdm(total=len(bbdd), unit="iteration", position=0)
            for bbdd_name, img in bbdd.items():
                desc = self.desc_function(img)
                desc_bbdd[bbdd_name] = desc

                progress_bar.update(1)

            with open(prep_result_path, 'wb') as file:
                pickle.dump(desc_bbdd, file)

        return desc_bbdd
