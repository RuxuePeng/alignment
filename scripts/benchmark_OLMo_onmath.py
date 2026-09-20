import json
import random

import numpy as np
import torch

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.modal_utils import app, image
from cs336_alignment.vllm_utils import VLLMServer

seed = 42
max_tokens = 512
gpu = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class BenchmarkBaseline:
    def __init__(self, model_id: str):
        # start inference engine
        self.inference_engine = VLLMServer(model_id=model_id, gpu=gpu, seed=seed)

    def prepare_questions(
        self, prompt_path: str, raw_questions: list[str]
    ) -> list[str]:
        # prepare questions and answers
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompts = [prompt_template.format(question=q) for q in raw_questions]
        return prompts

    def grade_answers(self, questions, llm_answers, gt_answers, reward_fn):
        grades = []
        correctness_category = {
            "both_correct": 0,
            "only_format_correct": 0,
            "neither_correct": 0,
        }
        for question, response, gt_answer in zip(questions, llm_answers, gt_answers):
            answer_only_response = response.text
            answer_only_groundt = gt_answer.split("####")[
                -1
            ]  # split out final answer from GSM8K dataset
            print("----------------------")
            print(f"[Sampled question] {question}\n")
            print(f"[OSS LLM answer] {answer_only_response}")
            print(f"[Ground truth answer] {answer_only_groundt}")

            grade: dict = reward_fn(
                response=answer_only_response, ground_truth=answer_only_groundt
            )
            if grade["format_reward"] and grade["answer_reward"]:
                correctness_category["both_correct"] += 1
            elif grade["format_reward"]:
                correctness_category["only_format_correct"] += 1
            else:
                correctness_category["neither_correct"] += 1

            print("[Grade the LLM answer using ground truth]")
            print(
                f"format reward: {grade['format_reward']}\n",
                f"answer reward: {grade['answer_reward']}\n",
                f"Reward: {grade['reward']}\n",
            )
            grades.append(grade["reward"])
        final_grade = sum(grades) / len(grades)
        print(f"Avg reward across sampled problems: {final_grade:.2f}\n")
        return final_grade, correctness_category

    def benchmark_question_only(self, sampled_data, benchmark_data_size: int) -> float:
        # 1. benchmark on question only
        questions, answers = sampled_data
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
            "max_tokens": max_tokens,  # how long each response will be
            "stop": "}}",  # stop when answer is finshed
            "include_stop_str_in_output": True,
        }
        completions: list = self.inference_engine.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=benchmark_data_size,
        )
        assert len(completions) == len(answers) == benchmark_data_size
        return self.grade_answers(
            prompts, completions, answers, question_only_reward_fn
        )

    def benchmark_0_shot(self, sampled_data, benchmark_data_size: int) -> float:
        # 2. benchmark on zero shot with question and instruction only
        questions, answers = sampled_data

        prompts = self.prepare_questions(
            prompt_path="cs336_alignment/prompts/r1_zero.prompt",
            raw_questions=questions,
        )

        # get response
        self.inference_engine.start()
        sampling_params = {
            "temperature": 1.0,  # temperature for the model, 1 means to use the raw distribution to sample words
            "n": 1,  # generate 1 respone per question
            "seed": seed,
            "max_tokens": max_tokens,  # how long each response will be
            "stop": "</answer>",  # stop when answer is finshed
            "include_stop_str_in_output": True,
        }
        completions: list = self.inference_engine.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=benchmark_data_size,
        )
        assert len(completions) == len(answers) == benchmark_data_size
        return self.grade_answers(prompts, completions, answers, r1_zero_reward_fn)

        # 3. benchmark on 3 shot

    def benchmark_few_shot(self, sampled_data, benchmark_data_size: int) -> float:
        # 3. benchmark on few shot, with examples = 3
        questions, answers = sampled_data

        prompts = self.prepare_questions(
            prompt_path="cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
            raw_questions=questions,
        )

        # get response
        self.inference_engine.start()
        sampling_params = {
            "temperature": 1.0,  # temperature for the model, 1 means to use the raw distribution to sample words
            "n": 1,  # generate 1 respone per question
            "seed": seed,
            "max_tokens": max_tokens,  # how long each response will be
            "stop": "</answer>",  # stop when answer is finshed
            "include_stop_str_in_output": True,
        }
        completions: list = self.inference_engine.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=benchmark_data_size,
        )
        assert len(completions) == len(answers) == benchmark_data_size
        return self.grade_answers(prompts, completions, answers, r1_zero_reward_fn)


def load_data(path: str):
    # read train.jsonl
    questions = []
    answers = []
    with open(path, "r") as f:
        for line in f:
            sample_dict = json.loads(line)
            questions.append(sample_dict["question"])
            answers.append(sample_dict["answer"])
    return questions, answers


def sample_data(data, sample_size):
    questions, answers = data
    indices = random.sample(range(len(answers)), sample_size)
    sampled_questions = [questions[i] for i in indices]
    sampled_answers = [answers[i] for i in indices]
    return sampled_questions, sampled_answers


@app.function(
    image=image,
    gpu="H200",
)
def benchmark() -> None:
    benchmark_sample_size = 20
    model_id = "allenai/OLMo-2-0425-1B-Instruct"
    data_path = "./data/gsm8k/train.jsonl"

    # prepare questions and answers
    questions, answers = load_data(data_path)
    sampled_questions, sampled_answers = sample_data(
        (questions, answers), sample_size=benchmark_sample_size
    )

    # use the same sampled data to benchmark 3 styles of prompting
    print("Benchmark question only style:")
    grade1, breakdown1 = BenchmarkBaseline(model_id=model_id).benchmark_question_only(
        sampled_data=(sampled_questions, sampled_answers),
        benchmark_data_size=benchmark_sample_size,
    )
    print("Benchmark 0 shot:")
    grade2, breakdown2 = BenchmarkBaseline(model_id=model_id).benchmark_0_shot(
        sampled_data=(sampled_questions, sampled_answers),
        benchmark_data_size=benchmark_sample_size,
    )
    # sample some examples to use
    print("Benchmark 3 shot:")
    grade3, breakdown3 = BenchmarkBaseline(model_id=model_id).benchmark_few_shot(
        sampled_data=(sampled_questions, sampled_answers),
        benchmark_data_size=benchmark_sample_size,
    )
    print(
        "Benchmark summary: \n",
        f"Model: {model_id}",
        f"dataset: {data_path}",
        f"Evaluation sample size: {benchmark_sample_size}\n"
        f"Question-only prompting: {grade1:.2f}, breakdown: {breakdown1}\n",
        f"Zero shot prompting: {grade2:.2f}, breakdown: {breakdown2}\n",
        f"Few shot prompting(Sample=3): {grade3:.2f}, breakdown: {breakdown3}\n",
    )


@app.local_entrypoint()
def main():
    benchmark.remote()
