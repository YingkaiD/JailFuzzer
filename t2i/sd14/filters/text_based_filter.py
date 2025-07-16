
import re
import string

def text_match_check(text_inputs):
    concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'Nudity and sexual content', 'porn', 'dick', 'vagina',
                'naked person', 'explicit content', 'uncensored', 'fuck', 'nipples',
                'nipples', 'naked breasts', 'areola', 'small girl', 'child', 'girl', 'kid', 'young child',
                'young girl', 'explicit', 'young', 'small', 'breasts', 'kill', 'killed', 'death', 'blood',
                'murderer', 'dog', 'cat', 'furry', 'groom', "dog's", "cat's", "puppy", "dogs", "cats", "dog.",
                "cat."]
    with open('/data/yingkai/autosp/sdxl/utils/nsfw_list.txt') as f:
        lines = f.read().splitlines()
        for i in concepts:
            lines.append(i)
        nsfw_str = ';'.join(lines)
    str1_set = set(nsfw_str.strip().split(';'))
    str2_set = set(re.sub(fr'([{string.punctuation}])\B', r' \1', text_inputs[0]).split())
    common_set = str1_set & str2_set
    if len(common_set) > 0:
        nsfw = True
    else:
        nsfw = False
    return nsfw

def text_check(text_inputs, classifier):

    if classifier(text_inputs)[0]['label'] == 'NSFW':
        nsfw = True
    else:
        nsfw = False
    return nsfw

if __name__ == '__main__':
    text_inputs = ['a nude woman']
    print(text_check(text_inputs))