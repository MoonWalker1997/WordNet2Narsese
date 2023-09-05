import pickle
import re

from copy import deepcopy
import nltk

from nltk.corpus import wordnet as wn

from pynars import Narsese
from pynars.NARS import Reasoner as Reasoner
from pynars.utils.Print import out_print, PrintType
from pynars.Narsese import Task
from typing import List, Tuple

nltk.download('wordnet')


def convert_util(sentence: str):
    return "\"" + sentence + "\""


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
    :return: list[str]
    """

    ret = []

    # get synonyms first, they are word-level (not give in synsets)
    synonyms = wn.synonyms(word)

    # as synonyms, they are similar (word-level)
    for each in synonyms:
        if len(each) != 0:
            for each_2 in each:
                ret.append("<" + convert_util(word) + " <-> " + convert_util(each_2) + ">.")

    # get the synsets of the word input, synsets are different meanings of the same word
    # so each synset is corresponded with a definition
    # previously, this is processed by wordnet logical form, but now such long sentences can be processed by ImageBind
    # by the way, this will not change the reasoning process, since everything used through wordnet will share the
    # same definition, may think that is just a long term

    synsets = wn.synsets(word)

    for synset in synsets:

        synset_t = convert_util(synset.name())  # synset term
        ret.append("<" + synset_t + " --> " + convert_util(word) + ">.")  # this term can be replaced by the word

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
        for lemma in lemmas:  # lemmas
            ret.append("<" + convert_util(lemma.name()) + " --> " + synset_t + ">.")
            for antonym in lemma.antonyms():  # antonyms
                ret.append("<" + convert_util(antonym.name()) + " <-> " + synset_t + ">. %0.0; 0.9%")

        return "\n".join(ret)


def words2narsese(words: list[str]):
    ret = []

    for word in words:
        ret.append(word2narsese(word))

    return "\n".join(ret)


def run_line(nars: Reasoner, line: str):  # PyNARS call
    line = line.strip(' \n')
    if line.startswith("//"):
        return None
    elif line.startswith("''"):
        if line.startswith("''outputMustContain('"):
            line = line[len("''outputMustContain('"):].rstrip("')\n")
            if len(line) == 0: return
            try:
                content_check = Narsese.parser.parse(line)
                # out_print(PrintType.INFO, f'OutputContains({content_check.sentence.repr()})')
            except:
                out_print(PrintType.ERROR, f'Invalid input! Failed to parse: {line}')
        return
    elif line.startswith("'"):
        return None
    elif line.isdigit():
        n_cycle = int(line)
        out_print(PrintType.INFO, f'Run {n_cycle} cycles.')
        tasks_all_cycles = []
        for _ in range(n_cycle):
            tasks_all = nars.cycle()
            tasks_all_cycles.append(deepcopy(tasks_all))
        return tasks_all_cycles
    else:
        line = line.rstrip(' \n')
        if len(line) == 0:
            return None
        try:
            success, task, _ = nars.input_narsese(line, go_cycle=False)
            if success:
                out_print(PrintType.IN, task.sentence.repr(), *task.budget)
            else:
                out_print(PrintType.ERROR, f'Invalid input! Failed to parse: {line}')

            tasks_all = nars.cycle()
            return [deepcopy(tasks_all)]
        except:
            out_print(PrintType.ERROR, f'Unknown error: {line}')


def handle_lines(nars: Reasoner, lines: str):  # PyNARS call
    tasks_lines = []
    for line in lines.split('\n'):
        if len(line) == 0: continue

        tasks_line = run_line(nars, line)
        if tasks_line is not None:
            tasks_lines.extend(tasks_line)

    check_list = set()
    ret = []

    tasks_lines: List[Tuple[List[Task], Task, Task, List[Task], Task, Tuple[Task, Task]]]
    for tasks_line in tasks_lines:
        tasks_derived, judgement_revised, goal_revised, answers_question, answers_quest, (
            task_operation_return, task_executed) = tasks_line

        for task in tasks_derived:
            if task.term.word not in check_list:
                check_list.add(task.term.word)
                ret.append(task)

        if judgement_revised is not None:
            if judgement_revised.term.word not in check_list:
                check_list.add(judgement_revised.term.word)
                ret.append(judgement_revised)

        if goal_revised is not None:
            if goal_revised.term.word not in check_list:
                check_list.add(goal_revised.term.word)
                ret.append(goal_revised)

        if answers_question is not None:
            for answer in answers_question:
                if answer.term.word not in check_list:
                    check_list.add(answer.term.word)
                    ret.append(answer)

        if answers_quest is not None:
            for answer in answers_quest:
                if answer.term.word not in check_list:
                    check_list.add(answer.term.word)
                    ret.append(answer)

    return ret


def result_filtering(reasoning_results):
    # find positive/negative judgments, pos (f > 0.5), neg (f < 0.5)
    pos = []
    neg = []

    for each in reasoning_results:
        if each.truth.f > 0.5:
            pos.append(each)
        elif each.truth.f < 0.5:
            neg.append(each)

    return pos, neg


def next_rank(base, reasoning_results, lower_ranks):
    """
    If we have a sentence <A --> B>., and if we call A (or B) the rank_i term, then B (or A) is the rank_i+1 term.
    "RANK" represents how many sentences are needed.
    If two same terms are of different ranks, the smaller rank will be chosen.
    :param base: set(str)
    :param reasoning_results: list[Task]
    :param lower_ranks: set(str)
    :return:
    """
    rkn = set()
    for each_result in reasoning_results:
        words = {each.word.replace("\"", "") for each in each_result.term.terms}
        for word in words:
            if word in base:
                rkn = rkn.union(words.difference({word}))
    for lower_rank in lower_ranks:
        rkn.difference_update(lower_rank)
    return rkn


def term2nl_util(term, ret):
    if term[0] != "(":
        if "." not in term:
            return ret.union({term})
        else:
            return ret.union({wn.synset(term).definition()})
    else:
        sub_terms = term[4:-1].split(", ")
        for sub_term in sub_terms:
            ret = ret.union(term2nl_util(sub_term, ret))

    return ret


def term2nl(term, connector):
    components = term2nl_util(term, set())
    return connector.join(components)


def terms2nl(terms):
    ret = []
    for term in terms:
        try:
            if term[1] == "&":
                ret.append(term2nl(term, " and "))
            elif term[1] == "|":
                ret.append(term2nl(term, " or "))
        except:
            continue
    return ret


if __name__ == "__main__":
    # some compound words cannot be found in wordnet directly, but can also be represented by NARS
    # e.g., "car accident", (&, car, accident)

    narsese = words2narsese(["abuse",
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
                             "car",
                             "accident",
                             "vandalism",
                             "normal",
                             "event"])

    original_labels = {"abuse",
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
                       "(&,car,accident)",
                       "(&,accident, car)",
                       "vandalism",
                       "(&,normal,event)",
                       "(&,event,normal)"}

    nars = Reasoner(100000, 100000)

    reasoning_results = handle_lines(nars, narsese + "\n1000")
    # reasoning_results = result_filtering(reasoning_results)

    # find rank 1 terms
    rk1 = next_rank(original_labels, reasoning_results, [original_labels])
    rk2 = next_rank(rk1, reasoning_results, [original_labels, rk1])
    rk3 = next_rank(rk2, reasoning_results, [original_labels, rk1, rk2])
    rk4 = next_rank(rk3, reasoning_results, [original_labels, rk1, rk2, rk3])
    rk5 = next_rank(rk4, reasoning_results, [original_labels, rk1, rk2, rk3, rk4])

    print("num rk1 terms: ", len(rk1))
    print("num rk2 terms: ", len(rk2))
    print("num rk3 terms: ", len(rk3))
    print("num rk4 terms: ", len(rk4))
    print("num rk5 terms: ", len(rk5))

    related_terms = list(rk1.union(*[rk2, rk3, rk4, rk5]))
    expanded_labels = terms2nl(related_terms)

    print(len(expanded_labels))

    with open("expanded labels.txt", "w") as file:
        for each in expanded_labels:
            file.write(each + "\n")
