from system_retrieval import filter, similarity, text, texture, mask, eval, histogram, keypoint

def create_histograms(choices):
    valid_choices = ['grayscale', 'threechannel', 'blocks', 'multilevel']

    histograms = []
    for hist_key, val in choices.items():
        bins = val[0]

        if hist_key == 'grayscale':
            print(f"Adding grayscale histogram with bins={bins}")
            hist_obj = histogram.GrayscaleHistogram(bins)
        elif hist_key == 'threechannel':
            print(f"Adding threechannel histogram with bins={bins}")
            hist_obj = histogram.ThreeChannelHistogram(bins)
        elif hist_key == 'blocks':
            block_size = val[1]

            print(f"Adding threechannel histogram with bins={bins}, block size={block_size}")
            hist_obj = histogram.HistogramWithBlocks(bins, block_size)
        elif hist_key == 'multilevel':
            levels = val[1:]
            print(f"Adding multilevel histogram with levels={levels}, bins={bins}")
            hist_obj = histogram.MultiLevelHistogram(bins, levels)
        else:
            raise ValueError(f"{hist_key} not recognized as a histogram type. Valid histogram types are: {valid_choices}")

        histograms.append(hist_obj)

    return histograms


def create_kp(type, **kwargs):
    if type[0] == 'surf':
        print(f"Using KP descriptor SURF")
        return keypoint.SURF_des()
 
    elif type[0] == 'sift':
        print(f"Using KP descriptor SIFT")
        return keypoint.SIFT_des()
    
    elif type[0] == 'orb':
        print(f"Using KP descriptor ORB")
        return keypoint.ORB_des()
    
    elif type[0] == 'fast':
        print(f"Using KP descriptor FAST")
        return keypoint.FAST_des()


    print("Create Keypoint descriptors {num_blocks}".format(num_blocks=type))



def create_filters(choices):
    valid_choices = ['median', 'average', 'maxrank', 'wmaxrank', 'wavelet']

    print("Create filters")

    filters = []
    for ft, val in choices.items():
        if ft == 'median':
            print(f"Adding median filter with neighborhood_size={val[0]}")
            filter_obj = filter.MedianFilter(val[0])
        elif ft == 'average':
            print(f"Adding average filter with ksize={val[0]}")
            filter_obj = filter.AverageFilter(val[0])
        elif ft == 'maxrank':
            print(f"Adding max rank filter with size_input={val[0]} and output_rank={val[1]}")
            filter_obj = filter.MaxRankFilter(val[0], val[1])
        elif ft == 'wmaxrank':
            print(f"Adding weighted max rank filter with size_input={val[0]} and output_rank={val[1]}")
            filter_obj = filter.WeightedMaxRankFilter(val[0], val[1])
        elif ft == 'wavelet':
            print(f"Adding wavelet filter")
            filter_obj = filter.WaveletFilter()
        else:
            raise ValueError(f"{ft} not recognized as a filter type. Valid filter types are: {valid_choices}")

        filters.append(filter_obj)

    return filters


def create_similarity(choice):
    print("Create similarity search or distance")

    if choice == 'histinter':
        similarity_obj = similarity.HistogramIntersection()
        print(f"Using Histogram Intersection similarity")
    elif choice == 'hellinger':
        similarity_obj = similarity.Hellinger()
        print(f"Using Hellinger similarity")
    elif choice == 'chi2':
        similarity_obj = similarity.Chi2Distance()
        print(f"Using Chi-Squared Distance similarity")
    elif choice == 'l1':
        similarity_obj = similarity.L1Distance()
        print(f"Using L1 Distance similarity")
    elif choice == 'l2texture':
        similarity_obj = similarity.L2DistanceTexture()
        print(f"Using L2 Distance similarity")
    elif choice == 'l1texture':
        similarity_obj = similarity.L1DistanceTexture()
        print(f"Using L1 Distance similarity")
    elif choice == 'levenshtein':
        similarity_obj = similarity.LevenshteinDistance()
        print(f"Using Levenshtein Distance similarity")
    elif choice == 'damlev':
        similarity_obj = similarity.DamerauLevenshteinDistance()
        print(f"Using Damerau-Levenshtein Distance similarity")
    elif choice == 'jaro':
        similarity_obj = similarity.JaroDistance()
        print(f"Using Jaro Distance similarity")
    elif choice == 'nw':
        similarity_obj = similarity.NwDistance()
        print(f"Using Needleman-Wunsch Distance similarity")
    elif choice == 'gotoh':
        similarity_obj = similarity.GotohDistance()
        print(f"Using Gotoh Distance similarity")
    elif choice == 'mlpins':
        similarity_obj = similarity.MlipnsDistance()
        print(f"Using MLIPNS Distance similarity")
    elif choice == 'hamming':
        similarity_obj = similarity.HammingDistanceWithPadding()
        print(f"Using Hamming Distance with Padding similarity")
    elif choice == 'knn':
        similarity_obj = similarity.knnDistance()
        print(f"Using KNN distance for KP")
    elif choice == 'l1kp':
        similarity_obj = similarity.L1kpDistance()
        print(f"Using L1 Distance for KP")
    elif choice == 'l2kp':
        similarity_obj = similarity.L2kpDistance()
        print(f"Using L2 Distance for KP")    
    elif choice == 'hammingkp':
        similarity_obj = similarity.HammingDistancekp()
        print(f"Using Hamming Distance for KP")    
    elif choice == 'flann':
        similarity_obj = similarity.FlannDistance()
        print(f"Using Flann Distance for KP")    
    else:
        raise ValueError(f"{choice} not recognized as a similarity or distance type")

    return similarity_obj


def create_mask():
    print("Create mask finder")
    return mask.MaskFinderTeam5()


def create_texture(type, **kwargs):
    if type == 'DCT':
        num_blocks = kwargs['num_blocks']
        N = kwargs['N']

        print("Create DCT descriptors for num_blocks {num_blocks} and {N}".format(num_blocks=num_blocks, N=N))

        return texture.DctDescriptor(num_blocks=num_blocks, N=N)


def create_eval(eval_dict):
    evaluators = []

    for type, args in eval_dict.items():
        if type == "mapk":
            for k in args:
                print(f"Create MAP with k={k}")
                evaluators.append(eval.MapkEvaluator(k))

    return evaluators
