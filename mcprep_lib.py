# Preprocessing of multiple-choice datasets

from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    load_from_disk,
    get_dataset_config_names,
    DatasetDict,
    Dataset,
)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass, field, fields
from tqdm import tqdm
import os
import re
import random
import logging
from logging import info, warn, error

logging.basicConfig(level=logging.INFO)


@dataclass
class DataCreationArguments:
    # model: str = field(
    #     default="meta-llama/Llama-2-7b-hf",
    #     metadata={"help": "The model to train, evaluate or use for predictions."},
    # )
    # args: TrainingArguments provided separately
    data: str = field(
        default="ai2_arc",
        metadata={
            "help": "Comma separated dataset:config:split triples to process (if config or split is unspecified all are processed)"
        },
    )
    exclude: str = field(
        default=None,
        metadata={
            "help": "Comma separated dataset:config:split triples to exclude (if config or split is unspecified all are excluded)"
        },
    )
    filter: bool = field(
        default=False,
        metadata={"help": "Filter questions with -100 labels."},
    )
    nexamples: int = field(
        default=0,
        metadata={
            "help": "Maximum number of q/a examples in one instance (0 means no limit)."
        },
    )
    # output_dir: str = field(
    #     default="mcprep_out",
    #     metadata={"help": "Directory to save the preprocessed dataset."},
    # )
    partial: bool = field(
        default=False,
        metadata={"help": "Split examples to fill window size."},
    )
    shuffle: bool = field(
        default=False,
        metadata={"help": "Shuffle the examples before grouping."},
    )
    window: int = field(
        default=2048,
        metadata={
            "help": "Maximum number of tokens in one instance (0 means no limit)."
        },
    )
    qtype: str = field(
        default="ai2_arc",
        metadata={"help": "Type of preprocessing"},
    )
    balanced: bool = field(
        default=False,
        metadata={"help": "Same number of correct and incorrect answers."},
    )
    cross_check_type: str = field(
        default="random",
        metadata={
            "help": "Whether or not to cross check samples with the scores given by a baseline. If used, must provide the path to the baseline scores."
        },
    )
    baseline_path: str = field(
        default=None,
        metadata={"help": "Path to the baseline scores."},
    )
    choice_source: str = field(
        default="default",
        metadata={"help": "Source of incorrect choices"},
    )
    choice_dataset: str = field(
        default=None,
        metadata={
            "help": "Dataset to draw incorrect choices from. Follows the same format as the data argument."
        },
    )
    dpo: bool = field(
        default=False,
        metadata={"help": "Save the output in DPO format (prompt, chosen, rejected)"},
    )

    def __post_init__(self):
        pass


# def parse_args():
#     parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         "-d",
#         "--data",
#         default="ai2_arc",
#         type=str,
#         help="Comma separated dataset:config:split triples to process (if config or split is unspecified all are processed)",
#     )
#     parser.add_argument(
#         "-e",
#         "--exclude",
#         default="",
#         type=str,
#         help="Comma separated dataset:config:split triples to exclude (if config or split is unspecified all are excluded)",
#     )
#     parser.add_argument(
#         "-f", "--filter", action="store_true", help="Filter questions with -100 labels."
#     )
#     parser.add_argument(
#         "-m",
#         "--model",
#         default="meta-llama/Llama-2-7b-hf",
#         type=str,
#         help="Model id for config and tokenizer",
#     )
#     parser.add_argument(
#         "-n",
#         "--nexamples",
#         default=0,
#         type=int,
#         help="Maximum number of q/a examples in one instance (0 means no limit).",
#     )
#     parser.add_argument(
#         "-o",
#         "--output_dir",
#         default="mcprep_out",
#         type=str,
#         help="Directory to save the preprocessed dataset.",
#     )
#     parser.add_argument(
#         "-p",
#         "--partial",
#         action="store_true",
#         help="Split examples to fill window size.",
#     )
#     parser.add_argument(
#         "-s",
#         "--shuffle",
#         action="store_true",
#         help="Shuffle the examples before grouping.",
#     )
#     parser.add_argument(
#         "-w",
#         "--window",
#         default=2048,
#         type=int,
#         help="Maximum number of tokens in one instance (0 means no limit).",
#     )
#     parser.add_argument(
#         "-t", "--qtype", default="ai2_arc", type=str, help="Type of preprocessing"
#     )
#     parser.add_argument(
#         "-b",
#         "--balanced",
#         action="store_true",
#         help="Same number of correct and incorrect answers.",
#     )
#     parser.add_argument(
#         "-x",
#         "--cross-check-type",
#         default="random",
#         type=str,
#         help="Whether or not to cross check samples with the scores given by a baseline. If used, must provide the path to the baseline scores.",
#         choices=["random", "second-best", "worst"],
#     )
#     parser.add_argument(
#         "-bp",
#         "--baseline-path",
#         default=None,
#         type=str,
#         help="Path to the baseline scores.",
#     )
#     parser.add_argument(
#         "-cs",
#         "--choice-source",
#         default="default",
#         type=str,
#         help="Source of incorrect choices",
#         choices=["default", "within", "otherdataset"],
#     )
#     parser.add_argument(
#         "-cd",
#         "--choice-dataset",
#         default=None,
#         type=str,
#         help="Dataset to draw incorrect choices from. Follows the same format as the data argument.",
#     )
#     parser.add_argument(
#         "-3",
#         "--dpo",
#         action="store_true",
#         help="Save the output in DPO format (prompt, chosen, rejected)",
#     )
#     args, unknown = parser.parse_known_args()
#     unknown and info(f"Warning: Unrecognized options: {unknown}")
#     info(args)
#     return args


def create_datasets(args):
    include = expand_triples(args.data)
    exclude = expand_triples(args.exclude) if args.exclude else {}
    choices = expand_triples(args.choice_dataset) if args.choice_dataset else {}
    args.triples = [key for key in include if key not in exclude]
    args.choices = [key for key in choices]
    if args.choice_source == "otherdataset":
        assert (
            len(args.choice_dataset) > 0
        ), "Must provide a dataset to draw incorrect choices from."

    if args.dpo:
        assert (
            args.cross_check_type == "random"
        ), "Cross-checking is not supported in DPO format yet."
        return dpoprep(args)
    else:
        if args.cross_check_type != "random":
            assert (
                args.baseline_path is not None
            ), "Must provide the path to the baseline scores."
        return mcprep(args)


def _load_dataset(path, name=None, split=None):
    if os.path.isdir(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, name=name, split=split, trust_remote_code=True)


def _load_baseline_scores(paths):
    import json

    paths = paths.split("|")
    paths = [x for x in paths if x]

    # Doc_id -> answertext -> score
    baseline_scores = {}
    for path in paths:
        with open(path, "r") as f:
            scores_dict = json.load(f)

            for score_dict in scores_dict:
                doc_id = score_dict["doc"]["id"]

                assert (
                    not doc_id in baseline_scores
                ), "Duplicate doc_id in baseline scores."

                baseline_scores[doc_id] = {}

                choices = score_dict["doc"]["choices"]["text"]
                score_tuples = score_dict["filtered_resps"]

                scores = [x[0] for x in score_tuples]

                for choice, score in zip(choices, scores):
                    baseline_scores[doc_id][choice] = score

    return baseline_scores


def expand_triples(speclist):
    triples = {}  # use dict instead of set to keep the insertion order
    for str in speclist.split(","):
        components = str.split(":")
        if len(components) == 3:  # dataset:config:split
            triples[(*components,)] = None
        elif len(components) == 2:  # dataset:config
            d = _load_dataset(
                *components, split=None
            )  # only loads dataset metadata according to chatgpt, not sure.
            for split in sorted(
                d.keys()
            ):  # unspecified configs and splits processed in sorted order.
                triples[(*components, split)] = None
        elif len(components) == 1:  # dataset only
            for config in sorted(get_dataset_config_names(*components)):
                d = _load_dataset(*components, config, split=None)
                for split in sorted(d.keys()):
                    triples[(*components, config, split)] = None
        else:
            warn(f"Skipping bad dataset spec {str}")
    return triples


def query(doc, qtype):
    if qtype == "ai2_arc":
        qtext = f"Question: {doc['question']}\nAnswer:"
    if qtype == "math":
        qtext = f"Problem: {doc['problem']}\nAnswer:"
    elif qtype == "hellaswag":
        qtext = (
            hellaswag_preprocess(doc["activity_label"])
            + ": "
            + doc["ctx_a"]
            + " "
            + doc["ctx_b"].capitalize()
        )
    elif qtype == "winogrande":
        pron = doc["sentence"].index("_")
        qtext = doc["sentence"][:pron].strip()
    elif qtype == "mmlu":
        keys = ["A", "B", "C", "D"]
        qtext = (
            doc["question"].strip()
            + "".join(
                [f"\n{key}. {choice}" for key, choice in zip(keys, doc["choices"])]
            )
            + "\nAnswer:"
        )
    elif qtype == "mmlu2":  # mmlu questions in arc format
        qtext = f"Question: {doc['question'].strip()}\nAnswer:"
    elif qtype == "ufeedback":  # Ultrafeedback questions in arc format
        qtext = f"Question: {doc['prompt'].strip()}\nAnswer:"
    else:
        error(f"{qtype} not recognized")
    return qtext


def choices(doc, qtype):
    if qtype == "ai2_arc":
        ans = doc["choices"]["text"]
    elif qtype == "hellaswag":
        ans = [hellaswag_preprocess(ending) for ending in doc["endings"]]
    elif qtype == "winogrande":
        pron = doc["sentence"].index("_")
        ans = [
            x + doc["sentence"][pron + 1 :] for x in [doc["option1"], doc["option2"]]
        ]
    elif qtype == "mmlu":
        ans = ["A", "B", "C", "D"]
    elif qtype == "mmlu2":
        ans = doc["choices"]
    elif qtype == "ufeedback":
        ans = [
            ufeedback_preprocess_choice(doc["chosen"]),
            ufeedback_preprocess_choice(doc["rejected"]),
        ]
    else:
        error(f"{qtype} not recognized")
    return ans


def gold(doc, qtype):
    if qtype == "ai2_arc":
        ans = doc["choices"]["label"].index(doc["answerKey"])
    elif qtype == "hellaswag":
        ans = int(doc["label"])
    elif qtype == "winogrande":
        ans = int(doc["answer"]) - 1
    elif qtype == "mmlu":
        ans = doc["answer"]
    elif qtype == "mmlu2":
        ans = doc["answer"]
    elif qtype == "ufeedback":
        ans = 0  # ufeedback_preprocess_choice(doc["chosen"])
    else:
        error(f"{qtype} not recognized")
    return ans


# from lm-evaluation-harness/lm_eval/tasks/hellaswag.py
def hellaswag_preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def ufeedback_preprocess_choice(dcts, num_skip=1):
    return "\n".join([x["content"] for x in dcts[num_skip:]])


def mcprep(args):
    tokenizer = args.tokenizer
    correct_dict = {
        "question": [],
        "answer": [],
    }  # keep q/a separate in case we use -100 labels
    incorrect_dict = {
        "question": [],
        "answer": [],
    }  # process both correct and incorrect answers, save as two datasets in a DatasetDict

    if args.choice_dataset is not None:
        alt_choices = []
        for dataset, config, split in args.choices:
            inferred_qtype = None
            if "arc" in dataset:
                inferred_qtype = "ai2_arc"
            elif "hellaswag" in dataset:
                inferred_qtype = "hellaswag"
            elif "mmlu" in dataset:
                inferred_qtype = "mmlu2"
            else:
                inferred_qtype = "ufeedback"

            dataset = _load_dataset(dataset, config)
            for doc in tqdm(dataset[split]):
                choices_list = choices(doc, inferred_qtype)
                alt_choices.extend(choices_list)
    else:
        alt_choices = None

    for dataset, config, split in args.triples:
        info(f"Processing {dataset}:{config}:{split}")
        dataset = _load_dataset(dataset, config)
        baseline_scores = (
            _load_baseline_scores(args.baseline_path)
            if args.cross_check_type != "random"
            else None
        )
        for doc in tqdm(dataset[split]):
            doc_id = doc.get("id", None)
            qtext = query(doc, args.qtype)
            qtoks = tokenizer.encode(f"\n\n{qtext}")[
                2:
            ]  # [2:] to skip the initial ['<s>', '▁']
            correct_index = gold(
                doc, args.qtype
            )  # doc['choices']['label'].index(doc['answerKey'])
            incorrect_count = 0

            if args.cross_check_type != "random":
                scores_dict = baseline_scores[doc_id]
                best_options = {}
                best_score = None

            if args.choice_source == "default":
                choices_list = choices(doc, args.qtype)
            elif args.choice_source == "within":
                # Randomly sample a document from the same dataset and split
                # and use its choices as the incorrect choices
                idx = random.randint(0, len(dataset[split]) - 1)
                default_choice_list = choices(doc, args.qtype)
                correct_choice = default_choice_list[correct_index]
                choices_list = choices(dataset[split][idx], args.qtype)
                choices_list = [x for x in choices_list if x != correct_choice]
                choices_list = [correct_choice] + choices_list
                correct_index = 0
                print("Changed {} to {}".format(default_choice_list, choices_list))
            elif args.choice_source == "otherdataset":
                # Randomly sample a document from the other dataset and split
                # and use its choices as the incorrect choices
                assert (
                    alt_choices is not None
                ), "Must provide a dataset to draw incorrect choices from."
                default_choice_list = choices(doc, args.qtype)
                correct_choice = default_choice_list[correct_index]
                num_wrong_choices = len(default_choice_list) - 1
                # Pick num_wrong_choices random choices from the other dataset
                choices_list = random.sample(alt_choices, num_wrong_choices)
                # choices_list = [correct_choice] + choices_list
                correct_index = 0
                print("Changed {} to {}".format(default_choice_list, choices_list))

            for index, atext in enumerate(choices_list):
                atoks = tokenizer.encode(atext)[
                    1:
                ]  # [1:] to skip the initial <s>, initial space automatically included
                ascore = (
                    scores_dict[atext] if args.cross_check_type != "random" else None
                )
                if index == correct_index:
                    qadict = correct_dict
                else:
                    if args.cross_check_type != "random":
                        if best_score is None:
                            best_score = ascore
                            best_options["question"] = qtoks
                            best_options["answer"] = atoks
                        else:
                            higher_better = args.cross_check_type == "second-best"

                            if higher_better == (ascore > best_score):
                                print(
                                    "Picking {} instead. Score changed from {} to {}".format(
                                        atext, best_score, ascore
                                    )
                                )
                                best_score = ascore
                                best_options["question"] = qtoks
                                best_options["answer"] = atoks

                            continue

                    if (
                        incorrect_count > 0
                        and args.balanced
                        and args.cross_check_type == "random"
                    ):
                        continue
                    qadict = incorrect_dict
                    incorrect_count += 1

                qadict["question"].append(qtoks)
                qadict["answer"].append(atoks)

            if args.cross_check_type != "random":
                incorrect_dict["question"].append(best_options["question"])
                incorrect_dict["answer"].append(best_options["answer"])

    input_ids = labels = nexamples = None

    def empty_input_ids():
        nonlocal input_ids, labels, nexamples
        input_ids = [tokenizer.bos_token_id]
        labels = [tokenizer.bos_token_id] if not args.filter else [-100]
        nexamples = 0

    for qadict in (correct_dict, incorrect_dict):
        qadict["input_ids"] = []
        qadict["labels"] = []
        qindex = list(range(len(qadict["question"])))
        if args.shuffle:
            random.shuffle(qindex)
        empty_input_ids()
        for i in qindex:
            q_i, a_i = qadict["question"][i], qadict["answer"][i]
            input_ids_i = q_i + a_i
            if args.window > 0 and not args.partial and len(input_ids_i) > args.window:
                continue  # skip example too long to fit window if no partial
            if args.filter:
                labels_i = [-100 for _ in q_i] + a_i
            else:
                labels_i = q_i + a_i
            if args.nexamples > 0 and nexamples == args.nexamples:
                qadict["input_ids"].append(input_ids)
                qadict["labels"].append(labels)
                empty_input_ids()
            while args.window > 0 and len(input_ids) + len(input_ids_i) > args.window:
                if args.partial:
                    p = args.window - len(input_ids)
                    input_ids.extend(input_ids_i[:p])
                    labels.extend(labels_i[:p])
                    input_ids_i = input_ids_i[p:]
                    labels_i = labels_i[p:]
                qadict["input_ids"].append(input_ids)
                qadict["labels"].append(labels)
                empty_input_ids()
            input_ids.extend(input_ids_i)
            labels.extend(labels_i)
            nexamples += 1
        if input_ids:
            qadict["input_ids"].append(input_ids)
            qadict["labels"].append(labels)

    return DatasetDict(
        {
            "dataset1": Dataset.from_dict(
                {
                    "input_ids": correct_dict["input_ids"],
                    "labels": correct_dict["labels"],
                }
            ),
            "dataset2": Dataset.from_dict(
                {
                    "input_ids": incorrect_dict["input_ids"],
                    "labels": incorrect_dict["labels"],
                }
            ),
        }
    )  # .save_to_disk(args.output_dir)


def dpoprep(args):
    dataset_dict = {"prompt": [], "chosen": [], "rejected": []}
    for dataset, config, split in args.triples:
        info(f"Processing {dataset}:{config}:{split}")
        dataset = _load_dataset(dataset, config)
        for doc in tqdm(dataset[split]):
            q = query(doc, args.qtype)
            g = gold(doc, args.qtype)
            c = choices(doc, args.qtype)
            i = random.randint(0, len(c) - 1)
            while i == g:
                i = random.randint(0, len(c) - 1)
            dataset_dict["prompt"].append(q + " ")
            dataset_dict["chosen"].append(c[g])
            dataset_dict["rejected"].append(c[i])
    Dataset.from_dict(dataset_dict).save_to_disk(args.output_dir)


__name__ == "__main__" and main()


# Notes:
# Sample tokenizer output for multiple questions ARC style:
# ['<s>', '▁Question', ':', '▁Which', '▁term', '▁is', '▁used', '▁to', '▁describe', '▁a', '▁physical', '▁property', '▁of', '▁a', '▁min', 'eral', '?', '<0x0A>', 'Answer', ':', '▁solid', '<0x0A>', '<0x0A>', 'Question', ':', '▁Small', '▁m', 'amm', 'als', '▁have', '▁many', '▁adapt', 'ations', '▁that', '▁keep', '▁them', '▁warm', '▁in', '▁winter', '.', '▁Which', '▁would', '▁not', '▁help', '▁con', 'serve', '▁heat', '?', '<0x0A>', 'Answer', ':', '▁running', '<0x0A>']
# tokenizer(text) returns dict with 'input_ids' and 'attention_mask'. tokenizer.encode(text) just returns the input_ids as list.
