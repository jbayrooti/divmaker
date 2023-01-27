import math
import numpy as np
import pytorch_lightning as pl
import nlp

FINETUNING_WARMUP_FRAC = 0.06  # From Roberta paper.

def get_lr_scheduler(config):
    if 'Pretrain' in config.system:
        wiki_percent = config.data_params.wikipedia_percentage
        wiki_dset = nlp.load_dataset("wikipedia", "20200501.en", split=f'train[:{wiki_percent}%]', cache_dir='/data5/atamkin/wikipedia')
        book_percent = config.data_params.bookcorpus_percentage
        book_dset = nlp.load_dataset("bookcorpus", split=f'train[:{book_percent}%]', cache_dir='/data5/atamkin/bookcorpus')
        examples_per_epoch = len(wiki_dset) + len(book_dset) 
        lr_scheduler = LRWarmupDecayCallback(
            learning_rate=config.optim_params.lr, 
            warmup_examples=1e5,
            total_examples=examples_per_epoch * config.num_epochs
        )
    elif 'Transfer' in config.system:
        if not config.task_name:
            raise RuntimeError('config.task_name must be specified')
        dset = nlp.load_dataset('glue', config.task_name, split='train')
        num_examples = len(dset)
        lr_scheduler = LRWarmupDecayCallback(
            learning_rate=config.optim_params.lr, 
            warmup_examples=np.floor(num_examples * FINETUNING_WARMUP_FRAC),
            total_examples=num_examples
        )
    return lr_scheduler

# For text models.
class LRWarmupDecayCallback(pl.Callback):

    def __init__(self, learning_rate, warmup_examples, total_examples, lr_decay=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.examples = 0
        self.lr_decay = lr_decay
        self.warmup_examples = warmup_examples
        self.total_examples = total_examples

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        optimizer = trainer.optimizers[0]
        
        # pretrain is index, input_ids, label_ids, pad_mask; transfer is index, txt, label = batch
        input_ids = batch[1]

        if self.lr_decay:
            self.examples += input_ids.size(0)
            if self.examples < self.warmup_examples:
                # linear warmup
                lr_mult = float(self.examples) / float(max(1, self.warmup_examples))
            else:
                # cosine learning rate decay
                progress = float(self.examples - self.warmup_examples) / float(
                    max(1, self.total_examples - self.warmup_examples))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * np.sign(param_group['lr'])  # Keep negative groups negative.
