import json
import random
import subprocess

import numpy as np
import torch

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.modal_utils import app, image, quote_command
from cs336_alignment.vllm_utils import VLLMServer

seed = 42
max_token = 512
gpu = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class BenchmarkBaseline:
    def __init__(self, model_id: str):
        # start inference engine
        self.inference_engine = VLLMServer(model_id=model_id, gpu=gpu, seed=seed)

    def load_data(self, path: str):
        # read train.jsonl
        questions = []
        answers = []
        with open(path, "r") as f:
            for line in f:
                sample_dict = json.loads(line)
                questions.append(sample_dict["question"])
                answers.append(sample_dict["answer"])
        return questions, answers

    def sample_data(self, data, sample_size):
        questions, answers = data
        indices = random.sample(range(len(answers)), sample_size)
        sampled_questions = [questions[i] for i in indices]
        sampled_answers = [answers[i] for i in indices]
        # print(sampled_questions[0], sampled_answers[0])
        return sampled_questions, sampled_answers

    def prepare_questions(
        self, prompt_path: str, raw_questions: list[str]
    ) -> list[str]:
        # prepare questions and answers
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompts = [prompt_template.format(question=q) for q in raw_questions]
        print(f"Example prompt: {prompts[0]}")
        return prompts

    def benchmark_question_only(self, benchmark_data_size: int, data_path: str):
        # 1. benchmark on question only
        # prepare questions and answers
        questions, answers = self.load_data(data_path)
        questions, answers = self.sample_data(
            (questions, answers), sample_size=benchmark_data_size
        )

        prompts = self.prepare_questions(
            prompt_path="cs336_alignment/prompts/question_only.prompt",
            raw_questions=questions,
        )
        # get response
        self.inference_engine.start()
        sampling_params = {
            "temperature": 1.0,  # temperature for the model, 1 means to use the raw distribution to sample words
            "n": 1,  # generate 1 respone per question
            "seed": seed,
            "max_token": max_token,  # how long each response will be
            "stop": "}}",  # stop when answer is finshed
            "include_stop_str_in_output": True,
        }
        completions: list = self.inference_engine.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=benchmark_data_size,
        )
        assert len(completions) == len(answers) == benchmark_data_size
        grades = []
        for response, answer in zip(completions, answers):
            answer_only_response = response.text
            answer_only_groundt = answer.split("####")[
                -1
            ]  # split out final answer from GSM8K dataset
            grade: dict = question_only_reward_fn(
                response=answer_only_response, ground_truth=answer_only_groundt
            )
            print(
                grade["format_reward"],
                grade["answer_reward"],
                grade["reward"],
            )
            grades.append(grade["reward"])
        print(f"Avg reward: {sum(grades) / len(grades):.4f}")

        # 2. benchmark on zero shot with question and instruction only

        # 3. benchmark on 3 shot


@app.function(
    image=image,
    gpu="T4",
)
def check_vllm() -> None:
    commands = [
        ["which", "vllm"],
        ["python", "-c", "import pyarrow; print('pyarrow:', pyarrow.__version__)"],
        ["python", "-c", "import datasets; print('datasets:', datasets.__version__)"],
        ["vllm", "--version"],
    ]

    for command in commands:
        print(f"$ {' '.join(command)}", flush=True)
        subprocess.run(command, check=False)


@app.local_entrypoint()
def main():
    check_vllm.remote()

    BenchmarkBaseline(
        model_id="allenai/OLMo-2-0425-1B-Instruct"
    ).benchmark_question_only(
        data_path="./data/gsm8k/train.jsonl",
        benchmark_data_size=10,
    )


# local testing
# main()
