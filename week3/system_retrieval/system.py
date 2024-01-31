import cv2
import numpy as np

from utils import utils
from . import text
from . import similarity
from . import descriptor
import pickle
from tqdm import tqdm
import random
import string
import os


class RetrievalSystem:
    def __init__(self, **kwargs):
        self.filters = kwargs["filters"]
        self.mask_finder = kwargs["mask_finder"]
        self.remove_bg = kwargs["remove_bg"]
        self.similarity_search = kwargs["similarity_search"]
        self.histograms = kwargs["histograms"]
        self.texture = kwargs["texture"]
        self.text = kwargs["text"]
        self.perform_ocr = True  # TODO: pass this as argument - do not init here
        self.text_reader = text.TextReader()
        self.text_finder = text.TextFinderWeek2()
        self.eval = kwargs["eval"]

    def preprocess_image(self, img):
        """

        Args:
            img: image to preprocess

        Returns:
            img - a list that contains one or two images, can be with or without background
            mask -
            bb_text - a list of lists that contains the coordinates of the bounding boxes of the text
        """
        # Filter
        for filter in self.filters:
            img = filter.filter(img)

        # Mask
        if self.mask_finder is not None:
            mask = self.mask_finder.find_mask(img)
        else:
            mask = None

        if self.remove_bg:
            # Creates a list of paintings
            img = utils.remove_background(mask, img)
  
        else:
            # For the sake of consistency put the image in a list
            img = [img]

                

        # Find text
        bb_text = []
        for i in img:
            bb_text.append(self.text_finder.find_text(i))

        return img, mask, bb_text

    def calculate_histograms(self, img, with_bbox=False, text_box=None):
        all_histogram = []
        for hist in self.histograms:

            h = hist.calculate_histogram(img, with_bbox, text_box)
            all_histogram.extend(h)

        return all_histogram


    def perform_retrieval(self, dsno):
        queries_path, bbdd_path, gt_path = "datasets/" + dsno, "datasets/BBDD", "datasets/" + dsno + "/gt_corresps.pkl"
        print(f"Query path: {queries_path}")
        print(f"BBDD path: {bbdd_path}")
        print(f"GT path: {gt_path}")

        delimiter = "-" * 40
        print(delimiter)

        queries = utils.import_images(queries_path)
        queries = sorted(queries.items())
        queries = dict(queries)
        self.queries = queries

        bbdd = utils.import_images(bbdd_path)
        bbdd = sorted(bbdd.items())
        bbdd = dict(bbdd)
        
        results_all = {}

        print("Start preprocessing the images")
        progress_bar = tqdm(total=len(queries), unit="iteration", position=0)

        prep_result_path = 'results/' + dsno + '_prep_result'
        if os.path.isfile(prep_result_path):
            with open(prep_result_path, 'rb') as file:
                queries = pickle.load(file)

            progress_bar.close()

        else:
            for query_name, query_img in queries.items():
                img, mask, bb_text = self.preprocess_image(query_img)
                queries[query_name] = img, mask, bb_text

                progress_bar.update(1)

            with open(prep_result_path, 'wb') as file:
                pickle.dump(queries, file)

        print("Finished preprocessing the images")

        if self.text:
            print("Calculate text-based descriptors")

            print("Read text from BBDD and keep authors")
            bbdd_text = utils.import_text(bbdd_path)
            bbdd_text = sorted(bbdd_text.items())
            bbdd_text = dict(bbdd_text)
            bbdd_text_author = {key: value[0] for key, value in bbdd_text.items()}

            desc_calculator = descriptor.TextDescriptor(
                ss_calculator=similarity.LevenshteinDistance(),
                desc_function=self.text_reader.read_text,
                dsno=dsno
            )
            result = desc_calculator.calculate(queries, bbdd_text_author)
            print(np.array(result).shape)
            results_all['text'] = result


        if self.histograms is not None and len(self.histograms) > 0:
            print("Calculate histogram-based descriptors")
           
            desc_calculator = descriptor.HistogramDescriptor(
                ss_calculator=similarity.HistogramIntersection(),
                desc_function=self.calculate_histograms,
                dsno=dsno
            )
            result = desc_calculator.calculate(queries, bbdd)
            print(np.array(result).shape)
            results_all['histo'] = result


        if self.texture is not None:
            print("Calculate texture-based descriptors")

            desc_calculator = descriptor.TextureDescriptor(
                ss_calculator=similarity.L2DistanceTexture(),
                desc_function=self.texture.calculate,
                dsno=dsno
            )
            result = desc_calculator.calculate(queries, bbdd)
            print(np.array(result).shape)
            results_all['texture'] = result


        if results_all:
            results_def = []
            print("Calculate the joint descriptor")

            lambda1 = 0.0 #text
            lambda2 = 0.1 #histograms
            lambda3 = 0.9 #texture

            if results_all['text'] and results_all['histo'] and results_all['texture']:
                final_results = []
                print(np.array(results_all['text']).shape)

                for i in range(0, 30):
                    result_per_query = {}

                    for k,v in results_all['text'][i].items():
                        if k not in result_per_query.keys():
                            result_per_query[k] = lambda1*v
                        else:
                            result_per_query[k] += lambda1*v

                    for k,v in results_all['histo'][i].items():
                        if k not in result_per_query.keys():
                            result_per_query[k] = lambda2*v
                        else:
                            result_per_query[k] += lambda2*v

                    for k,v in results_all['texture'][i].items():
                        if k not in result_per_query.keys():
                            result_per_query[k] = lambda3*v
                        else:
                            result_per_query[k] += lambda3*v

                    best_k = dict(sorted(result_per_query.items(), key=lambda item: item[1], reverse=True))
                    ll = [utils.extract_number_from_string(key) for key in best_k.keys()]

                    final_results.append(ll)


                for evaluator in self.eval:
                    gt = pickle.load(open(gt_path, 'rb', buffering=0), encoding='bytes')

                    result_eval = evaluator.evaluate(gt, final_results, dsno)
                    print(result_eval)
           



