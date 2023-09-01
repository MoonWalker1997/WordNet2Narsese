import pickle
import re

import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def convert_util(sentence: str):
    return sentence.replace("-", "_")


def sentence2term(sentence):
    """
    This function is to change a sentence in natural language (with spaces inside) to a term can be used in Narsese.
    Symbols in this sentence will be replaced.
    :param sentence: str
    :return: str
    """
    tmp = list(filter(None, re.split("\W", sentence)))

    return "_".join(tmp)


def word2narsese(word: str):
    """
    This function is to change a single word into many Narsese sentences for reasoning.
    :param word: str
    :return: list[str]
    """

    ret = []

    # get synonyms first, they are word-level (not give in synsets)
    synonyms = wn.synonyms(word)

    # as synonyms, they are similar (word-level)
    for each in synonyms:
        if len(each) != 0:
            for each_2 in each:
                ret.append("<" + word + " <-> " + convert_util(each_2) + ">.")

    # get the synsets of the word input, synsets are different meanings of the same word
    # so each synset is corresponded with a definition
    # previously, this is processed by wordnet logical form, but now such long sentences can be processed by ImageBind
    # by the way, this will not change the reasoning process, since everything used through wordnet will share the
    # same definition, may think that is just a long term

    synsets = wn.synsets(word)

    for synset in synsets:

        synset_t = synset.name()  # synset term
        ret.append("<" + synset_t + " --> " + word + ">.")  # this term can be replaced by the word

        for each in synset.hypernyms():  # hypernyms
            ret.append("<" + synset_t + " --> " + convert_util(each.name()) + ">.")
        for each in synset.hyponyms():  # hyponyms
            ret.append("<" + convert_util(each.name()) + " --> " + synset_t + ">.")
        for each in synset.instance_hypernyms():  # instance hypernyms
            ret.append("<{" + synset_t + "} --> " + convert_util(each.name()) + ">.")
        for each in synset.instance_hyponyms():  # instance hyponyms
            ret.append("<{" + convert_util(each.name()) + "} --> " + synset_t + ">.")
        for each in synset.member_holonyms():  # member holonyms
            ret.append("<(*," + convert_util(each.name()) + "," + synset_t + ") --> MemberOf>.")
        for each in synset.substance_holonyms():  # substance holonyms
            ret.append("<(*," + convert_util(each.name()) + "," + synset_t + ") --> SubstanceOf>.")
        for each in synset.part_holonyms():  # part holonyms
            ret.append("<(*," + convert_util(each.name()) + "," + synset_t + ") --> PartOf>.")
        for each in synset.member_meronyms():  # member meronyms
            ret.append("<(*," + synset_t + "," + convert_util(each.name()) + ") --> MemberOf>.")
        for each in synset.substance_meronyms():  # substance meronyms
            ret.append("<(*," + synset_t + "," + convert_util(each.name()) + ") --> SubstanceOf>.")
        for each in synset.part_meronyms():  # part meronyms
            ret.append("<(*," + synset_t + "," + convert_util(each.name()) + ") --> PartOf>.")
        for each in synset.attributes():  # attributes
            ret.append("<(&," + convert_util(each.name()) + "," + synset_t + ") --> " + synset_t + ">.")
        for each in synset.entailments():  # entailments
            ret.append("<" + convert_util(each.name()) + " =/> " + synset_t + ">.")
        for each in synset.causes():  # causes
            ret.append("<" + synset_t + " =/> " + convert_util(each.name()) + ">.")
        for each in synset.also_sees():  # also sees
            ret.append("<" + synset_t + " <|> " + convert_util(each.name()) + ">.")
        for each in synset.verb_groups():  # verb groups
            ret.append("<" + synset_t + " <-> " + convert_util(each.name()) + ">.")
        for each in synset.similar_tos():  # similar-to's
            ret.append("<" + synset_t + " <-> " + convert_util(each.name()) + ">.")

        lemmas = synset.lemmas()
        for lemma in lemmas:
            ret.append("<" + lemma.name() + " --> " + synset_t + ">. %0.7; 0.9%")

        return "\n".join(ret)


def words2narsese(words: list[str]):
    ret = []

    for word in words:
        ret.append(word2narsese(word))

    return "\n".join(ret)


if __name__ == "__main__":
    print(words2narsese(["abuse",
                         "burglary",
                         "robbery",
                         "stealing",
                         "shooting",
                         "shoplifting",
                         "assault",
                         "fighting",
                         "arson",
                         "explosion",
                         "arrest",
                         "accident",
                         "vandalism",
                         "normal"]))
