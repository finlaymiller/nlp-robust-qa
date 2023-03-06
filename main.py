import argparse
from pathlib import Path
import os
import time
import json
import torch
import numpy as np
from tqdm.auto import tqdm
import collections
import evaluate
from datasets import load_dataset
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, TrainingArguments, Trainer, DefaultDataCollator


ms = "distilbert-base-uncased"
pad_on_right = False
tokenizer = None

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")

    parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed")
    parser.add_argument("--train", action="store", default="squad", help="Dataset to finetune on")
    parser.add_argument("--test", action="store", default="squad", help="Dataset to test on")
    parser.add_argument("--save", action="store_true", help="Save model")

    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args

def preprocess_duorc(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    for i, offset in enumerate(inputs.pop("offset_mapping")):
        answer = answers[i]
        if not isinstance(answer, str):
            if len(answer):
                answer = answer[0]
            else:
                answer = ""
        start_char = examples["context"][i].find(answer)
        end_char = examples["context"][i].find(answer) + len(answer)
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    return inputs

def preprocess_squad(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    for i, offset in enumerate(inputs.pop("offset_mapping")):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # print(f"\nq: {questions[i]}\na: {answer}\nc: {start_char}\t{end_char}\nx: {context_start}\t {context_end}\ns: {sequence_ids}")

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

def prepare_validation_features(examples):
    global tokenizer
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=384,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer
        predictions[example["id"]] = best_answer["text"]

    return predictions

# NOTE: Training pipeline adapted from https://huggingface.co/docs/transformers/tasks/question_answering
def main(raw_args=None):
    st = time.time()
    global ms
    global tokenizer
    global pad_on_right

    # parameters
    n_best_size = 20
    max_answer_length = 30

    # quiet some error messages
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = get_args(raw_args)

    # load the dataset
    if args.train == "squad":
        dataset = load_dataset("squad", split="train")
    elif args.train == "duorc":
        dataset = load_dataset("duorc", "SelfRC", split="train")

    if args.test == "squad":
        val_dataset = load_dataset("squad", split="validation")
    elif args.test == "duorc":
        val_dataset = load_dataset("duorc", "SelfRC", split="validation")
    
    # rename columns to align with squad names
    if args.train == "duorc":
        dataset = dataset.rename_column("plot", "context")
        dataset = dataset.rename_column("question_id", "id")
    if args.test == "duorc":
        val_dataset = val_dataset.rename_column("plot", "context")
        val_dataset = val_dataset.rename_column("question_id", "id")

    # load pretrained model and tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(ms)
    model = DistilBertForQuestionAnswering.from_pretrained(ms)
    pad_on_right = tokenizer.padding_side == "right"

    # tokenize the data
    ppf = preprocess_squad if args.train == "squad" else preprocess_duorc
    tokenized_dataset = dataset.map(ppf, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=dataset.column_names)
    ppfv = preprocess_squad if args.test == "squad" else preprocess_duorc
    tokenized_dataset_v = val_dataset.map(ppfv, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=val_dataset.column_names)

    # initialize training arguments
    training_args_dict = {
        "output_dir": ".outputs", # for intermediary checkpoints
        "save_strategy": "epoch" if args.save else "no",
        "do_train": args.train is not None,
        "do_eval": True,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "fp16": True,
        "deepspeed": "ds_config_zero_1.json" if args.deepspeed else None,
    }
    training_args = TrainingArguments(**training_args_dict)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset.select(range(10000)),
        eval_dataset=tokenized_dataset_v,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    # train
    if args.train:
        train_result = trainer.train()

        # extract performance metrics
        train_metrics = train_result.metrics
        train_metrics["train_samples"] = len(tokenized_dataset)
        trainer.log_metrics("train", train_metrics)

        # save trained model
        if args.save:
            trained_model_folder = f"{ms}-finetuned-{args.train}"
            trained_model_path = Path(trained_model_folder)
            trained_model_path.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(trained_model_path / "weights")

    # eval
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_dataset_v)
    trainer.log_metrics("eval", eval_metrics)

    batch = None
    for batch in trainer.get_eval_dataloader():
        break
    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
    with torch.no_grad():
        output = trainer.model(**batch)

    validation_features = val_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))

    start_logits = output.start_logits[0].cpu().numpy()
    end_logits = output.end_logits[0].cpu().numpy()
    offset_mapping = validation_features[0]["offset_mapping"]
    # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
    # an example inde
    context = val_dataset[0]["context"]

    # Gather the indices the best start/end logits:
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue
            if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    }
                )

    valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
    example_id_to_index = {k: i for i, k in enumerate(val_dataset["id"])}
    features_per_example = collections.defaultdict(list)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset]

    for i, feature in enumerate(validation_features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    final_predictions = postprocess_qa_predictions(val_dataset, validation_features, raw_predictions.predictions)

    # export to files for evaluation
    if args.test != "squad":
        from eval_custom import load_eval
        da = "duorc_answers.json"
        dp = "duorc_predictions.json"

        answers = [{"qa": references}]
        predictions = {}
        for k, v in final_predictions.items():
            predictions[k] = v

        with open(da, "w") as f:
            json.dump(answers, f)
            f.close()

        with open(dp, "w") as f:
            json.dump(predictions, f)
            f.close()

        res = load_eval(da, dp, args.test)
    else:
        metric = evaluate.load(args.test)
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        res = metric.compute(predictions=formatted_predictions, references=references)

    print(f"Finished evaluating {ms} on {args.test} (finetuned on {args.train}) in {round(time.time() - st)}s", res)

if __name__ == "__main__":
    main()
