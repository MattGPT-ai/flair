from typing import Iterable, Tuple, Union, NamedTuple

from flair.data import Corpus, MultiCorpus
from flair.models import MultitaskModel
from flair.nn import Classifier, Model


class TaskConfig(NamedTuple):
    model: Classifier
    corpus: Corpus
    task_id: str = ""
    loss_factor: float = 1.0


def make_multitask_model_and_corpus(
    mapping: Iterable[TaskConfig]
) -> Tuple[Model, Corpus]:
    models = []
    corpora = []
    loss_factors = []
    ids = []

    for task_i, task_map in enumerate(mapping):
        models.append(task_map[0])
        corpora.append(task_map[1])
        if task_map[2]:
            ids.append(task_map[2])
        else:
            ids.append(f"Task_{task_i}")
        loss_factors.append(task_map[3])

    return MultitaskModel(models=models, task_ids=ids, loss_factors=loss_factors), MultiCorpus(corpora, ids)
