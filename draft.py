from copy import deepcopy
from nltk.corpus import wordnet as wn
from typing import Tuple, Union
from pathlib import Path
from pynars import Narsese, NAL, NARS
from time import sleep
from multiprocessing import Process
import os
from pynars.Narsese.Parser.parser import TreeToNarsese
from pynars.Narsese import Sentence
import random
from pynars.NARS import Reasoner as Reasoner
from pynars.utils.Print import out_print, PrintType
from pynars.Narsese import Task
from typing import List
from pynars.utils.tools import rand_seed
import argparse


def info(title):
    print(f'''
============= {title} =============
module name: {__name__}
parent process: {os.getppid()}
process id: {os.getpid()}
============={'=' * (len(title) + 2)}=============
    ''')


def run_line(nars: Reasoner, line: str):
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


def handle_lines(nars: Reasoner, lines: str):
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
            ret.append(term2nl(term))
        except:
            continue
    return ret

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Parse NAL files.')
    # parser.add_argument('filepath', metavar='Path', type=str, nargs='*',
    #                     help='file path of an *.nal file.')
    # args = parser.parse_args()
    # filepath: Union[list, None] = args.filepath
    # filepath = filepath[0] if (filepath is not None and len(filepath) > 0) else None
    #
    # seed = 314159
    # rand_seed(seed)
    # out_print(PrintType.COMMENT, f'rand_seed={seed}', comment_title='Setup')
    # nars = Reasoner(10000, 10000)
    # out_print(PrintType.COMMENT, 'Init...', comment_title='NARS')
    # out_print(PrintType.COMMENT, 'Run...', comment_title='NARS')
    # # console
    # out_print(PrintType.COMMENT, 'Console.', comment_title='NARS')
    # lines = "<A --> B>.\n<B --> C>.\n10"
    # ret = handle_lines(nars, lines)
    # print(1)

    print(terms2nl(["(&, dog.n.01, kunnan)"]))
