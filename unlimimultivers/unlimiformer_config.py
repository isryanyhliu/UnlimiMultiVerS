from typing import Any

class UnlimiformerArguments:
    def __init__(self):
        self.unlimiformer_layer_begin = -1
        self.unlimiformer_layer_end = None
        self.unlimiformer_head_num = None
        self.unlimiformer_exclude_attention = False
        self.unlimiformer_max_len = None
        self.unlimiformer_chunk_overlap = 0.0
        self.unlimiformer_verbose = False
        self.tokenizer = None

    @staticmethod
    def add_arguments_to_parser(parser):
        parser.add_argument("--unlimiformer_layer_begin", type=int, default=-1, help="Layer to begin applying Unlimiformer.")
        parser.add_argument("--unlimiformer_layer_end", type=int, default=None, help="Layer to end applying Unlimiformer.")
        parser.add_argument("--unlimiformer_head_num", type=int, default=None, help="Specific head number for Unlimiformer.")
        parser.add_argument("--unlimiformer_exclude_attention", action="store_true", help="Exclude attention.")
        parser.add_argument("--unlimiformer_max_len", type=int, default=None, help="Max length for Unlimiformer.")
        parser.add_argument("--unlimiformer_chunk_overlap", type=float, default=0.0, help="Chunk overlap for Unlimiformer.")
        parser.add_argument("--unlimiformer_verbose", action="store_true", help="Verbose output for Unlimiformer.")
        parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer for Unlimiformer.")
        return parser
