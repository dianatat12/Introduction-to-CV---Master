import cv2
import numpy as np

from utils import utils
from . import text
from . import similarity
from . import descriptor
from . import keypoint
import pickle
from tqdm import tqdm
import random
import string
from PIL import Image


class RetrievalSystem:
    def __init__(self, **kwargs):
        self.filters = kwargs["filters"]
        self.mask_finder = kwargs["mask_finder"]
        self.remove_bg = kwargs["remove_bg"]
        self.similarity_search = kwargs["similarity_search"]
        self.histograms = kwargs["histograms"]
        self.texture = kwargs["texture"]
        self.text = kwargs["text"]
        self.kp = kwargs["kp"]
        self.perform_ocr = True  # TODO: pass this as argument - do not init here
        self.text_reader = text.TextReader()
        self.text_finder = text.TextFinderWeek2()
        self.eval = kwargs["eval"]

    def denoise_and_sharpen(self,image, kernel_size):

        denoised = cv2.medianBlur(image, 3)
        denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

        return sharpened
        
    def denoise_image(self,image, threshold):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        patch_height = height // 15
        patch_width = width // 15
        noisy_pixel_threshold = threshold

        noisy_regions = []
        clean_regions = []

        for y in range(0, height - patch_height + 1, patch_height):
            for x in range(0, width - patch_width + 1, patch_width):
                patch = gray_image[y:y+patch_height, x:x+patch_width]
                std_dev = patch.std()
                if std_dev > noisy_pixel_threshold:
                    noisy_regions.append((x, y, std_dev))
                else:
                    clean_regions.append((x, y, std_dev))

        if clean_regions:
            return image
        else:
            return self.denoise_and_sharpen(image, kernel_size=3)

    def preprocess_image(self, img):

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


    def perform_retrieval(self, queries_path, bbdd_path, gt_path):
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
        iteration = 0
        for query_name, query_img in queries.items():
            img, mask, bb_text = self.preprocess_image(query_img)
            queries[query_name] = img, mask, bb_text # can be a list of stuff
            progress_bar.update(1)

            
            """for i, elem in enumerate(img):
                name = str(iteration) + "old_filter_newmask.jpg"
                cv2.imwrite(name, elem)
                iteration += 1"""

                    

        print("Finished preprocessing the images")

        if self.text:
            print("Calculate text-based descriptors")

            print("Read text from BBDD and keep authors")
            bbdd_text = utils.import_text(bbdd_path)
            bbdd_text = sorted(bbdd_text.items())
            bbdd_text = dict(bbdd_text)
            bbdd_text_author = {key: value[0] for key, value in bbdd_text.items()}
            print(bbdd_text_author)

            desc_calculator = descriptor.TextDescriptor(
                ss_calculator=similarity.LevenshteinDistance(),
                desc_function=self.text_reader.read_text
            )
            result= desc_calculator.calculate(queries, bbdd_text_author)
            results_all['text'] = result

            print("Evaluate")
            gt = pickle.load(open(gt_path, 'rb', buffering=0), encoding='bytes')
            print("GT", gt)
            for evaluator in self.eval:
                result_eval = evaluator.evaluate(gt, result)
                print(result_eval)


        if self.histograms is not None and len(self.histograms) > 0:
            print("Calculate histogram-based descriptors")
           
            desc_calculator = descriptor.HistogramDescriptor(
                ss_calculator=similarity.HistogramIntersection(),
                desc_function=self.calculate_histograms
            )
            result = desc_calculator.calculate(queries, bbdd)
            results_all['histo'] = result

            print("Evaluate")
            gt = pickle.load(open(gt_path, 'rb', buffering=0), encoding='bytes')
            print(np.array(gt).shape)

            for evaluator in self.eval:
                result_eval = evaluator.evaluate(gt, result)
                print(result_eval)


        if self.texture is not None:
            print("Calculate texture-based descriptors")

            desc_calculator = descriptor.TextureDescriptor(
                ss_calculator=self.similarity_search,
                desc_function=self.texture.calculate
            )
            result= desc_calculator.calculate(queries, bbdd)
            results_all['texture'] = result

            print("Calculate similarity/distance")
            gt = pickle.load(open(gt_path, 'rb', buffering=0), encoding='bytes')
            for evaluator in self.eval:
                result_eval = evaluator.evaluate(gt, result)
                print(result_eval)
                

        if self.kp is not None:
            print("Calculate kp-based descriptors")

            desc_calculator = descriptor.KPDescriptor(
                ss_calculator=self.similarity_search,
                desc_function=self.kp.calculate
            )
            result= desc_calculator.calculate(queries, bbdd)
            results_all['kp'] = result

            print("Calculate similarity/distance")
            gt = pickle.load(open(gt_path, 'rb', buffering=0), encoding='bytes')
            print(gt)
            for evaluator in self.eval: 
                result_eval = evaluator.evaluate(gt, result)
                print(result_eval)
        
        if results_all and self.kp is None:
            results_def = []
            print("Calculate the joint descriptor")

            lambda1 = 0.0 #text
            lambda2 = 0.1 #histograms
            lambda3 = 0.9 #texture

            if results_all['text'] and results_all['histo'] and results_all['texture']:
                final_results = []
                for i in range(0, 30):
                    final_partial = []
                    print(len(results_all['text'][i]))
                    for j in range(len(results_all['text'][i])):
                        result_per_query = {}

                        for k,v in results_all['text'][i][j].items():
                            if k not in result_per_query.keys():
                                result_per_query[k] = lambda1*v
                            else:
                                result_per_query[k] += lambda1*v

                        for k,v in results_all['histo'][i][j].items():
                            if k not in result_per_query.keys():
                                result_per_query[k] = lambda2*v
                            else:
                                result_per_query[k] += lambda2*v

                        for k,v in results_all['texture'][i][j].items():
                            if k not in result_per_query.keys():
                                result_per_query[k] = lambda3*v
                            else:
                                result_per_query[k] += lambda3*v

                        best_k = dict(sorted(result_per_query.items(), key=lambda item: item[1], reverse=True)[:10])
                        ll = [utils.extract_number_from_string(key) for key in best_k.keys()]

                        final_partial.append(ll)
                    final_results.append(final_partial)
                print(final_results)
                print("lo de sobre en teoria es lo bo")


        
        """
        
        if results_all:
            results_def = []
            print("Calculate the joint descriptor")

            lambda1 = 0#text
            lambda2 = 0#histograms
            lambda3 = 1#texture
            value_k = 10

            
            if results_all['text'] and results_all['histo'] and results_all['texture']:
                for t,h,tt  in zip(results_all['text'], results_all['histo'], results_all['texture']):
                    print("aqui tenim t que hauria de ser una llista de 2", len(t))
                    print(t)
                    new_results = {}
                    result_partial = []
                    
                    for i in range(len(t)):
                        for k,v in t[i].items():
                            if k not in new_results.keys():
                                new_results[k] = lambda1*v
                            else:
                                new_results[k] += lambda1*v

                        for k,v in h[i].items():
                            if k not in new_results.keys():
                                new_results[k] = lambda2*v
                            else:
                                new_results[k] += lambda2*v

                        for k,v in tt[i].items():
                            if k not in new_results.keys():
                                new_results[k] = lambda3*v
                            else:
                                new_results[k] += lambda3*v
                        

                        best_k = dict(sorted(new_results.items(), key=lambda item: item[1], reverse=True)[:value_k])
                        ll = [utils.extract_number_from_string(key) for key in best_k.keys()]
                        result_partial.append(ll)
                    results_def.append(result_partial)
            print(results_def)

            for evaluator in self.eval:
                    result_eval = evaluator.evaluate(gt, results_def)
                    print(result_eval)
            """
           



