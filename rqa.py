import os
import csv
import json
import time
import util
import torch
from tqdm import tqdm
from trainer import Trainer
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='qa')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='datasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='datasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='datasets/oodomain_test')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)

    args = parser.parse_args()

    return args


def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples


def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'

    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
        print(len(tokenized_examples))
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples


def get_dataset(args, datasets, data_dir, tokenizer, split_name):
    dataset_dict = None
    dataset_name=''

    for dataset, num_samples in datasets.items():
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')

        for k, v in dataset_dict_curr.items():
            dataset_dict_curr[k] = v[:num_samples]
            print(f"Using {num_samples}/{len(v)} samples from {dataset}")

        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)

    data_encodings = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)

    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict


def main():
    st = time.time()
    args = get_args()
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    log = util.get_logger(args.save_dir, 'log_train')
    train_datasets = {"squad": 3333, "nat_questions": 3333, "newsqa": 3333}
    # val_datasets = {"squad": 4000, "nat_questions": 4000, "newsqa": 4000}
    val_datasets = {"duorc": 400, "race": 400, "relation_extraction": 400}
    args.eval_dir = "datasets/oodomain_val"

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        
        log.info(f'Args: {json.dumps(vars(args), indent=2, sort_keys=True)}')

        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        train_dataset, train_dict = get_dataset(args, train_datasets, args.train_dir, tokenizer, 'train')

        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, val_datasets, args.eval_dir, tokenizer, 'val')

        # print(f"train dataset keys: ", train_dataset.keys)
        # print(f"train dict keys: ", train_dict.keys())
        # print(f"vak dataset keys: ", val_dataset.keys)
        # print(f"vak dict keys: ", val_dict.keys())

        # tad = {
        # "output_dir": args.save_dir,
        # "save_strategy": "epoch" if args.save_dir else "no",
        # "do_train": args.do_train,
        # "do_eval": args.do_eval,
        # "per_device_train_batch_size": 16,
        # "per_device_eval_batch_size": 16,
        # "learning_rate": args.lr,
        # "num_train_epochs": args.num_epochs,
        # "weight_decay": 0.01,
        # "fp16": True,
        # "deepspeed": "ds_config_zero_1.json" if args.deepspeed else None,
        # }

        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        log.info("TRAINING")
        trainer = Trainer(args, log)
        

        train_result = trainer.train(model, train_loader, val_loader, val_dict)

        log.info("FINISHED TRAINING")
        # extract performance metrics
        # train_metrics = train_result.metrics
        # train_metrics["train_samples"] = sum(train_datasets.values())
        # trainer.log_metrics("train", train_metrics)
    
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        # log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, val_datasets, args.eval_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        # sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        # log.info(f'Writing submission file to {sub_path}...')
        # with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        #     csv_writer = csv.writer(csv_fh, delimiter=',')
        #     csv_writer.writerow(['Id', 'Predicted'])
        #     for uuid in sorted(eval_preds):
        #         csv_writer.writerow([uuid, eval_preds[uuid]])

    log.info(f"TIME {round(time.time() - st)}s")


if __name__ == "__main__":
    main()