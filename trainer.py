import os
import util
import torch
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device
        model.eval()
        all_start_logits = []
        all_end_logits = []

        with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)

                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                data_loader.dataset.encodings,
                                                (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        # best_scores = {'F1': -1.0, 'EM': -1.0}
        best_scores = [{ 'step': 0, 'F1': 0.0, 'EM': 0.0}]
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()

                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)

                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())

                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)

                        self.log.info(f'Eval {results_str}')

                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                                        
                        # if curr_score['F1'] >= best_scores['F1']:
                        #     best_scores = curr_score
                        #     self.save(model)
                        best_scores.append({'step': global_idx, 'F1': curr_score['F1'], 'EM': curr_score['EM']})
                        
                    global_idx += 1
        

        self.log.info(f'Final evaluation...')
        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())

        for k, v in curr_score.items():
            tbx.add_scalar(f'val/{k}', v, global_idx)

        self.log.info(f'Eval {results_str}')

        if self.visualize_predictions:
            util.visualize(tbx,
                            pred_dict=preds,
                            gold_dict=val_dict,
                            step=global_idx,
                            split='val',
                            num_visuals=self.num_visuals)
                        
        best_scores.append({'step': global_idx, 'F1': curr_score['F1'], 'EM': curr_score['EM']})
        self.save(model)

        return best_scores
