import argparse

ss_choices = ['histinter', 'hellinger', 'chi2', 'l1', 'l1texture', 'l2texture', 'levenshtein', 'damlev', 'jaro', 'nw', 'gotoh',
              'mlpins', 'hamming']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', nargs='+')
    parser.add_argument('-f', '--filter', nargs='+', required=False)
    parser.add_argument('-m', '--mask', action='store_true')
    parser.add_argument('-rb', '--removebackground', action='store_true', default=False)
    parser.add_argument('-t', '--text', action='store_true', default=False)
    parser.add_argument('-tx', '--texture', nargs='+')
    parser.add_argument('-hi', '--histogram', nargs='+')
    parser.add_argument('-ss', '--similarity_search', choices=ss_choices)
    parser.add_argument('-ev', '--eval', nargs='+', required=True)

    return parser.parse_args()


def create_dict_with_parameters(args_list):
    result = {}

    for item in args_list:
        if item.isalpha():
            current_word = item
            result[current_word] = []
        elif current_word is not None:
            result[current_word].append(int(item))

    return result

def parse_histogram(args_list):
    histograms = create_dict_with_parameters(args_list)

    return histograms

def parse_filters(args_list):
    return create_dict_with_parameters(args_list)

def parse_path(args_list):
    if len(args_list) < 3:
        raise ValueError("Please pass the path for the queries, bbdd, and gt or none at all ")

    query_path = args_list[0]
    bbdd_path = args_list[1]
    gt_path = args_list[2]

    return query_path, bbdd_path, gt_path


def parse_texture(args_list):
    print(args_list)
    if args_list[0]:
        type = args_list[0]
    else:
        type = 'DCT'

    if args_list[1].isnumeric():
        num_blocks = args_list[1]
    else:
        num_blocks = 4

    if args_list[2].isnumeric():
        N = args_list[2]
    else:
        N = 100

    return type, int(num_blocks), int(N)


def parse_eval(args_list):
    return create_dict_with_parameters(args_list)

