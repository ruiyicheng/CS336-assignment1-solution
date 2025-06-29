import regex as re

def experimental_bpe(text : str, num_of_merge : int ) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    vocabulary = ["<|endoftext|>"]+[chr(i) for i in range(256)]
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
        print(token_presentation)
        print(vocabulary)





def init_vocabulary():
    pass
def pre_tokenize():
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path : str, vocab_size : int, special_tokens : list[str]) -> tuple[dict[int, bytes],list[tuple[bytes, bytes]]]:
    pass

experimental_bpe("low low low low low lower lower widest widest widest newest newest newest newest newest newest",6)