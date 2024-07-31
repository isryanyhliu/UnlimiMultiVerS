from typing import Any

class UnlimiformerArguments:
    def __init__(self):
        self.test_unlimiformer = True
        self.unlimiformer_verbose = False
        self.unlimiformer_layer_begin = 0
        self.unlimiformer_layer_end = None
        self.unlimiformer_head_num = None
        self.unlimiformer_exclude_attention = False
        self.unlimiformer_max_len = None
        self.unlimiformer_chunk_overlap = 0.5
        self.unlimiformer_chunk_size = None
        self.random_unlimiformer_training = False
        self.unlimiformer_training = False
        self.use_datastore = False
        self.flat_index = False
        self.test_datastore = False
        self.reconstruct_embeddings = False
        self.gpu_datastore = True
        self.gpu_index = True
        self.tokenizer = None

    @staticmethod
    def add_arguments_to_parser(parser):
        parser.add_argument("--test_unlimiformer", type=bool, default=True, help="Whether to use Unlimiformer.")
        parser.add_argument("--unlimiformer_verbose", type=bool, default=False, help="Verbose output for Unlimiformer.")
        parser.add_argument("--unlimiformer_layer_begin", type=int, default=0, help="Layer to begin applying Unlimiformer.")
        parser.add_argument("--unlimiformer_layer_end", type=int, default=None, help="Layer to end applying Unlimiformer.")
        parser.add_argument("--unlimiformer_head_num", type=int, default=None, help="Specific head number for Unlimiformer.")
        parser.add_argument("--unlimiformer_exclude_attention", type=bool, default=False, help="Exclude attention.")
        parser.add_argument("--unlimiformer_max_len", type=int, default=None, help="Max length for Unlimiformer.")
        parser.add_argument("--unlimiformer_chunk_overlap", type=float, default=0.5, help="Chunk overlap for Unlimiformer.")
        parser.add_argument("--unlimiformer_chunk_size", type=int, default=None, help="Size of each input chunk.")
        parser.add_argument("--random_unlimiformer_training", type=bool, default=False, help="Random Unlimiformer training.")
        parser.add_argument("--unlimiformer_training", type=bool, default=False, help="Unlimiformer during training.")
        parser.add_argument("--use_datastore", type=bool, default=False, help="Use a datastore for Unlimiformer.")
        parser.add_argument("--flat_index", type=bool, default=False, help="Use a flat index for the datastore.")
        parser.add_argument("--test_datastore", type=bool, default=False, help="Test the datastore.")
        parser.add_argument("--reconstruct_embeddings", type=bool, default=False, help="Reconstruct embeddings in the datastore.")
        parser.add_argument("--gpu_datastore", type=bool, default=True, help="Use GPU for the datastore.")
        parser.add_argument("--gpu_index", type=bool, default=True, help="Use GPU for the index.")
        parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer for Unlimiformer.")
        return parser
