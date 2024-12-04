from mmengine.config import read_base
from opencompass.models import HuggingFaceNoiseModel

with read_base():
    from opencompass.configs.datasets.inference_ppl.inference_ppl import inference_ppl_datasets

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

workdir = 'outputs/inference_ppl'
datasets = [*inference_ppl_datasets]

models = []
for x in [1.0, 1,4, 2.2]:
    std = 0.1 * x
    models.append(dict(
        type=HuggingFaceNoiseModel,
        abbr=f'llama-3-8b-Instruct-std-{std}',
        path='meta-llama/Llama-3.1-8B-Instruct',
        max_out_len=128,
        generation_kwargs= {"noise_std": std},
        batch_size=32,
        #run_cfg=dict(num_gpus=1),
    ))


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=256,  # Maximum concurrent evaluation task count
    ),
)


# -------------Evaluation Stage ----------------------------------------
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
        max_num_workers=256,
    )
)
