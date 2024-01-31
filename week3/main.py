from system_retrieval import parser, system, ffactories

args = parser.parse_args()

queries_path, bbdd_path, gt_path = "/home/user/Documents/MASTER/C1/datasets/qsd2_w3", "/home/user/Documents/MASTER/C1/datasets/BBDD", "/home/user/Documents/MASTER/C1/datasets/qsd2_w3/gt_corresps.pkl"
filters = []
mask_finder = None
histograms, texture, text = None, None, None
ss, eval = None, None

if args.path:
    dsno = args.path[0]
else:
    dsno = 'qsd1_w3'

if args.filter:
    parsed = parser.parse_filters(args.filter)
    filters = ffactories.create_filters(parsed)

# TODO: Add parameters?
if args.mask:
    mask_finder = ffactories.create_mask()

remove_background = args.removebackground

if args.text is not None and args.text == True:
    print("Create text descriptor calculator")
    text = True

if args.histogram:
    print("Create histogram descriptor calculator")
    if 'with_bbox' in args.histogram and text == None:
        text = text.ffactories.create_text()

    hist_parsed = parser.parse_histogram(args.histogram)
    histograms = ffactories.create_histograms(hist_parsed)


if args.texture:
    print("Create texture descriptor calculator")
    type, num_blocks, N = parser.parse_texture(args.texture)
    texture = ffactories.create_texture(type, num_blocks=num_blocks, N=N)

if args.similarity_search:
    ss = ffactories.create_similarity(args.similarity_search)

if args.eval:
    print(args.eval)
    parsed = parser.parse_eval(args.eval)
    eval = ffactories.create_eval(parsed)

retrieval_sytem = system.RetrievalSystem(
    filters=filters,
    mask_finder=mask_finder,
    remove_bg=remove_background,
    histograms=histograms,
    texture=texture,
    text=text,
    similarity_search=ss,
    eval=eval
)

retrieval_sytem.perform_retrieval(dsno=dsno)
