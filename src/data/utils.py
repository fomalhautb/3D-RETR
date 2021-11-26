from torch.utils.data.dataloader import default_collate


def normalize(x):
    return x * 2 - 1


def denormalize(x):
    return (x + 1) / 2


class Collator:
    def __init__(self, tokenizer, max_length=30):
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, batch):
        batch = default_collate(batch)
        text = batch['text']
        tokenized = self._tokenizer(
            text,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True,
            max_length=self._max_length
        )

        batch['raw_text'] = text
        batch['text'] = tokenized['input_ids']
        batch['text_mask'] = tokenized['attention_mask'].bool()

        return batch
