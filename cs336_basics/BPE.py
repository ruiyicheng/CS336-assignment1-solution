import regex as re
import os
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool
import tqdm
import cProfile
import pstats
import pickle
from typing import Dict, List, Tuple, Iterable,Iterator
import time

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def experimental_bpe(text : str, num_of_merge : int ) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    vocabulary = []
    text_split = text.split(' ')
    wc_dict = {}
    for word in text_split:
        if word not in wc_dict:
            wc_dict[word] = 0
        wc_dict[word] += 1
    print(wc_dict)
    token_presentation = {}
    for key in wc_dict:
        token_presentation[tuple([key[i] for i in range(len(key))])] = wc_dict[key]


    for iter in range(num_of_merge):
        pair_stat_dict = {}
        maximum_count = 0
        for key in token_presentation:
            for i in range(len(key)-1):
                pair = (key[i],key[i+1])
                if pair not in pair_stat_dict:
                    pair_stat_dict[pair] = 0
                pair_stat_dict[pair] += token_presentation[key]
                if pair_stat_dict[pair] > maximum_count:
                    maximum_count = pair_stat_dict[pair]
        
        longest_pair_candidate_list = []
        for pair in pair_stat_dict:
            if pair_stat_dict[pair] == maximum_count:
                longest_pair_candidate_list.append(pair)
        pair_merge_this = max(longest_pair_candidate_list)
        new_token = pair_merge_this[0] + pair_merge_this[1]
        vocabulary.append(new_token)
        token_presentation_new = {}

        for key in token_presentation:
            key_list_merged = []
            merged = 0
            for i in range(len(key)):
                if merged:
                    merged = 0
                    continue
                if i == len(key)-1:
                    key_list_merged.append(key[i])
                    continue
                pair = (key[i],key[i+1])
                if pair_merge_this == pair:
                    key_list_merged.append(new_token)
                    merged = 1
                else:
                    key_list_merged.append(key[i])
            token_presentation_new[tuple(key_list_merged)] = token_presentation[key]

        token_presentation = token_presentation_new
    return 





def init_vocabulary():
    pass
def pre_tokenize():
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenization(path_dataset : str,start : int, end : int, special_tokens_for_re:list[str]) -> dict[str,int]:
    with open(path_dataset, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    splited_text_list = re.split('|'.join(special_tokens_for_re),chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    wc_dict = {}
    #print(end - start)
    #print(len(splited_text_list))
    for splited_text_this in splited_text_list:
        # do tokenization
        words = re.finditer(PAT,splited_text_this)
        
        for word in words:
            if word.group() not in wc_dict:
                wc_dict[word.group()] = 0
            wc_dict[word.group()] += 1

    #print(wc_dict)
    return wc_dict

def pre_tokenization_wrapper(args):
    return pre_tokenization(*args)

def reduce_pre_tokenization(result_dict_collected : list[dict[str,int]]) -> dict[str,int]:
    result_dict = defaultdict(int)
    for d in result_dict_collected:
        for k,v in d.items():
            result_dict[k] += v
    return dict(result_dict)
def train_bpe(input_path : str, vocab_size : int, special_tokens : list[str], parallelize_pre_token = True) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:

        

    
    # init vocabulary list
    vocab_list = [i.encode('utf-8') for i in special_tokens]+[bytes([i]) for i in range(256)]
    #print(vocab_list)
    special_tokens_for_re = []
    for special_token in special_tokens:
        list_components = special_token.split('|')
        special_tokens_for_re.append('\\|'.join(list_components))
    #print(special_tokens_for_re)
    number_of_chunk = 10
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, number_of_chunk, "<|endoftext|>".encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.

    if not parallelize_pre_token:
        result_dict_collected = []
        for start, end in tqdm.tqdm(zip(boundaries[:-1], boundaries[1:])):
            result_dict_collected.append(pre_tokenization(input_path,start,end,special_tokens_for_re))
        reduced_dict = reduce_pre_tokenization(result_dict_collected)
    else:
        import time
        t0 = time.time()
        number_of_core = 10
        #print([(input_path,start,end,special_tokens_for_re) for start,end in zip(boundaries[:-1], boundaries[1:])])
        with Pool(number_of_core) as p:
            result_dict_collected = p.map(pre_tokenization_wrapper, [(input_path,start,end,special_tokens_for_re) for start,end in zip(boundaries[:-1], boundaries[1:])])
        reduced_dict = reduce_pre_tokenization(result_dict_collected)
        print(f'Per-tokenization taken {time.time()-t0}s with {number_of_core} pool')
    token_presentation = {}
    for key in reduced_dict:
        # save the token presentation in a tuple of bytes

        token_presentation[tuple(bytes([i]) for i in key.encode('utf-8'))] = reduced_dict[key]
    
    #print(token_presentation)
    #print(reduced_dict)
    print(len(token_presentation))

    pair_stat_dict = {}
    maximum_count = 0
    for key in token_presentation:
        for i in range(len(key)-1):
            pair = (key[i],key[i+1])
            if pair not in pair_stat_dict:
                pair_stat_dict[pair] = 0
            pair_stat_dict[pair] += token_presentation[key]
            if pair_stat_dict[pair] > maximum_count:
                maximum_count = pair_stat_dict[pair]

    merges = []
    for iter in tqdm.tqdm(range(vocab_size-len(vocab_list))):
        # pair_stat_dict = {}
        # maximum_count = 0
        # for key in token_presentation:
        #     for i in range(len(key)-1):
        #         pair = (key[i],key[i+1])
        #         if pair not in pair_stat_dict:
        #             pair_stat_dict[pair] = 0
        #         pair_stat_dict[pair] += token_presentation[key]
        #         if pair_stat_dict[pair] > maximum_count:
        #             maximum_count = pair_stat_dict[pair]
        
        # longest_pair_candidate_list = []
        # for pair in pair_stat_dict:
        #     if pair_stat_dict[pair] == maximum_count:
        #         longest_pair_candidate_list.append(pair)
        best_pair = max(pair_stat_dict, key=pair_stat_dict.get)
        max_count = pair_stat_dict[best_pair]
        best_pairs = [p for p, count in pair_stat_dict.items() if count == max_count]
        pair_merge_this = max(best_pairs)
        new_token = pair_merge_this[0] + pair_merge_this[1]
        merges.append((pair_merge_this[0],pair_merge_this[1]))

        vocab_list.append(new_token)
        token_presentation_new = defaultdict(int)

        for key in token_presentation:
            freq_key = token_presentation[key]
            if not any(key[i:i+2]==pair_merge_this for i in range(len(key)-1)):
                token_presentation_new[key] = token_presentation_new.get(key, 0) + freq_key
                continue
            else:
                
                
                for j in range(len(key)-1):
                    pair = (key[j],key[j+1])
                    pair_stat_dict[pair] -= freq_key
                    if pair_stat_dict[pair] <= 0:
                        del pair_stat_dict[pair]

                key_list_merged = []
                i = 0
                while i < len(key):
                    if i == len(key)-1:
                        key_list_merged.append(key[i])
                        i+=1
                        continue
                    pair = (key[i],key[i+1])
                    if pair_merge_this == pair:
                        key_list_merged.append(new_token)
                        i += 2
                    else:
                        key_list_merged.append(key[i])
                        i += 1
                for j in range(len(key_list_merged)-1):
                    pair = (key_list_merged[j],key_list_merged[j+1])
                    if pair not in pair_stat_dict:
                        pair_stat_dict[pair] = 0
                    pair_stat_dict[pair] += freq_key
                token_presentation_new[tuple(key_list_merged)] += token_presentation[key]


        token_presentation = token_presentation_new
    vocab_list_ret = {i:vocab_list[i] for i in range(len(vocab_list))}
    return vocab_list_ret,merges
                
            
Vocab = Dict[int, bytes]
Merges = List[Tuple[bytes, bytes]]

def save_tokenizer(vocab: Vocab, merges: Merges, filepath_vocab: str, filepath_merger: str) -> None:
    """
    Save vocab and merges to a file using pickle.
    """
    with open(filepath_vocab, 'wb') as f:
        pickle.dump(vocab, f)
    with open(filepath_merger, 'wb') as f:
        pickle.dump(merges, f) 

def load_tokenizer(filepath_vocab: str,filepath_merger:str) -> tuple[Vocab, Merges]:
    """
    Load vocab and merges from a file using pickle.
    """
    with open(filepath_vocab, 'rb') as f:
        vocab = pickle.load(f)
    with open(filepath_merger, 'rb') as f:
        merge = pickle.load(f)   
    
    return vocab, merge
#experimental_bpe("low low low low low lower lower widest widest widest newest newest newest newest newest newest",6)

class Tokenizer:
    def __init__(self,vocab: Vocab,merges:Merges,special_tokens:list[str] | None = None):
        if special_tokens is None:
            special_tokens = []
        special_tokens = sorted(special_tokens, key=len, reverse=True) # make the longer pattern first
        ct = 0
        vocab_list = [v for k,v in vocab.items()]
        lenvoc = len(vocab_list)
        self.vocab = vocab
        max_cache_size = 10 # GB
        for special_token in special_tokens:
            bytes_special_token = special_token.encode('utf-8')
            
            if not bytes_special_token in vocab_list:
                self.vocab[ct+lenvoc] = bytes_special_token
                ct += 1
        #print(self.vocab)
        self.reverse_token_vocab_dict = {}
        for k,v in self.vocab.items():
            self.reverse_token_vocab_dict[v] = k
                


        
        self.merges = merges
        self.special_tokens = special_tokens

        self.special_tokens_for_re = []
        for special_token in special_tokens:
            list_components = special_token.split('|')
            self.special_tokens_for_re.append('\\|'.join(list_components))
        re_expression_special = '|'.join(self.special_tokens_for_re)
        self.special_regex = re.compile(re_expression_special)

        self.word_cache = {}
        
        
    @classmethod
    def from_files(cls,vocab_filepath : str, merges_filepath : str, special_tokens: list[str] | None =None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merged = pickle.load(f)   
        return cls(vocab,merged,special_tokens = special_tokens)
    
    def encode(self, text: str) -> list[int]:
        def process_text_this(text_this,token_res):
            if text_this['special_token']:
                token_res.append(self.reverse_token_vocab_dict[text_this['content'].encode('utf-8')])
                return
            else:
                words = re.finditer(PAT,text_this['content'])
                
                for word in words:
                    
                    word = word.group()
                    if word in self.word_cache:
                        token_res.extend(self.word_cache[word])
                        continue
                    presentation_all_words = []
                    presentation_map_dict = {}                    
                    splitted_chr = []
                    for i in range(len(word)): 
                        encoded_this = word[i].encode('utf-8') 
                        # if encoded_this in self.reverse_token_vocab_dict:
                        #     #character already in vocab
                        #     splitted_chr.append(encoded_this)
                        # else:
                        #     #character not in vocab, represent in raw bytes (which must have)
                        for bt in encoded_this:
                            splitted_chr.append(bytes([bt]))

                    splitted_chr = tuple(splitted_chr)
                    presentation = splitted_chr
                    presentation_all_words.append(presentation)
                    
                    presentation_map_dict[splitted_chr] = splitted_chr

                    #print(presentation_map_dict)
                    for mg in self.merges:
                        # presentation_map_dict_new = presentation_map_dict
                        for k,v in presentation_map_dict.items():
                            i = 0
                            merged_v = []
                            while i < len(v):
                                if i == len(v)-1:
                                    merged_v.append(v[i])
                                    i+=1
                                    continue
                                #print((v[i],v[i+1]),mg)
                                if (v[i],v[i+1])==mg:
                                    merged_v.append(v[i]+v[i+1])
                                    i += 2
                                    #print('merged',mg,'for',v)
                                else:
                                    merged_v.append(v[i])
                                    i += 1
                            presentation_map_dict[k] = tuple(merged_v)
                        # presentation_map_dict = presentation_map_dict_new
                    
                    
                    presentation_map_dict_new = {}
                    for k,v in presentation_map_dict.items():
                        v_list_this = []
                        for i in v:
                            v_list_this.append(self.reverse_token_vocab_dict[i])
                        presentation_map_dict_new[k] = v_list_this
                    ret_list = []
                    for present in presentation_all_words:
                        ret_list.extend(presentation_map_dict_new[present])
                    
                    token_res.extend(ret_list)
                    self.word_cache[word] = ret_list
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # words = re.finditer(PAT,text)
        text_pattern_result = []
        last_end = 0
        print('pre-tokenizing text')
        token_res = []
        if len(self.special_tokens_for_re) > 0:
            for match in self.special_regex.finditer(text):
                start, end = match.span()
                matched_str = match.group(0)
                trial_str = text[last_end:start]
                if len(trial_str)>0:
                    #text_pattern_result.append({'content':trial_str, 'special_token': False})
                    process_text_this({'content':trial_str, 'special_token': False},token_res)
                #text_pattern_result.append({'content':matched_str, 'special_token': True})
                process_text_this({'content':matched_str, 'special_token': True},token_res)
                # print(start,end)
                # print(matched_str)
                last_end = end
        trial_str = text[last_end:]
        if len(trial_str)>0:
            #text_pattern_result.append({'content':trial_str, 'special_token': False})
            process_text_this({'content':trial_str, 'special_token': False},token_res)
        #print(text_pattern_result)
        
        #for text_this in text_pattern_result:
            
                    #print(self.word_cache)
        #print(token_res)
        return token_res





    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_paragraph in iterable:
            for i in self.encode(text_paragraph):
                yield i
    
    def decode(self, ids: list[int]) -> str:
        byte_undefined_chr = '\uFFFD'.encode('utf-8')
        byte_stream = ''.encode('utf-8')
        for id_this in ids:
            if id_this in self.vocab:
                byte_stream += self.vocab[id_this]
            else:
                byte_stream += byte_undefined_chr
        return byte_stream.decode('utf-8',errors='replace')

if __name__=='__main__':

    vocab_path = 'tokenizer_vocab_tinystory.pkl'
    merge_path = 'tokenizer_merge_tinystory.pkl'
    path_dataset = "../data/TinyStoriesV2-GPT4-train.txt"
    # number_of_chunk = 100
    # path_dataset = "../tests/fixtures/tinystories_sample_5M.txt"
    vocabulary_size = 10000
    special_tokens = ["<|endoftext|>"]
    #cProfile.run('train_bpe(path_dataset,number_of_chunk,special_tokens)','profile_results.prof')
    #p = pstats.Stats('profile_results.prof')
    #p.sort_stats('cumtime').print_stats()
    vocab,merges = train_bpe(path_dataset,vocabulary_size,special_tokens)
    save_tokenizer(vocab, merges, vocab_path,merge_path)
    
    # loaded_vocab, loaded_merges = load_tokenizer('tokenizer_vocab.pkl','tokenizer_merge.pkl')
    # print(loaded_vocab)
    # print(loaded_merges)
    # longest_length = -1
    # longest_v = ''
    # for k,v in loaded_vocab.items():
    #     if len(v)>longest_length:
    #         longest_length = len(v)
    #         longest_v = v
    # print(f'longest vocabulary is {longest_v} with length {longest_length}')
    # tk = Tokenizer({0: b' ', 1: b'a', 2:b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),(b' a', b't')],special_tokens=['<|endoftext|>','<|endoftext|><|endoftext|>'])
    # token_res = tk.encode('<|endoftext|><|endoftext|>a acth at ate<|endoftext|><|endoftext|> the<|endoftext|> the')
    # print(token_res)
    tk = Tokenizer.from_files(vocab_path,merge_path,special_tokens = '<|endoftext|>')
    #datapath = "/Users/ruiyicheng/Documents/code/python/CS336/CS336-assignment1-solution/tests/fixtures/tinystories_sample.txt"
    #datapath = "/Users/ruiyicheng/Documents/code/python/CS336/CS336-assignment1-solution/tests/fixtures/tinystories_sample_5M.txt"
    datapath = "/Users/ruiyicheng/Documents/code/python/CS336/CS336-assignment1-solution/data/TinyStoriesV2-GPT4-train.txt"
    # compression ratio of encoder
    number_of_chunk = 1000
    with open(datapath, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, number_of_chunk, "<|endoftext|>".encode("utf-8"))
    print("Encoding with tokenizer from files...")
    encoded_all = []
    # with open(datapath, 'rb') as f:
    #     print('reading file',datapath)
    #     for start, end in tqdm.tqdm(zip(boundaries[:-1], boundaries[1:])):
        
    #         f.seek(start)
    #         text = f.read(end - start).decode("utf-8", errors="ignore")+"<|endoftext|>"
    #         #print(text)
    #         #print('file length',len(text))
    #         t0 = time.time()
    #         encoded = tk.encode(text)
    #         # encode the text per 100MB to save memory
    #         # encoded = tk.encode_iterable(tqdm.tqdm(text.splitlines()))
    #         dt = time.time() - t0
    #         print("Compression ratio:", len(text.encode('utf-8'))/len(encoded) )
    #         # print("Encoding time:", dt)
    #         print("Encode Bps:", len(text.encode('utf-8'))/dt)
    #         print('taken',dt,'s to encode',len(text.encode('utf-8')),'bytes')
    #         encoded_all.extend(encoded)
    # a parallelized version of the encoded data
    n_cores = 8
    print(f"Encoding with tokenizer from files in parallel with {n_cores} cores...")
    with Pool(n_cores) as p:
        with open(datapath, 'rb') as f:
            encoded = list(tqdm.tqdm(p.imap(tk.encode, [f.read(end - start).decode("utf-8", errors="ignore")+"<|endoftext|>" for start, end in zip(boundaries[:-1], boundaries[1:])]), total=len(boundaries)-1))
    savepath = "tokenized_tinystories_train.npy"
    import numpy as np
    print(f"Saving encoded data to {savepath}...")
    encoded_save = []
    for enc in encoded:
        encoded_save.extend(enc)
    encoded_save = np.array(encoded_save, dtype=np.uint16)  # Use uint16 for smaller size
    print(encoded_save.shape)
    np.save(savepath, encoded_save)
