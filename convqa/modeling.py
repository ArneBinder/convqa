import copy

from torch import nn
from torch.autograd import Function
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel, GPT2Model, PretrainedConfig
from transformers.modeling_utils import SequenceSummary


class GradReverse(Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class GPT2MultiHeadsAdversarialClModel(GPT2PreTrainedModel):
    r"""
        **mc_token_ids**: (`optional`, default to index of the last token of the input) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        **mc_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        **cl_labels_0, cl_labels_1, ...**: (`optional`) ``torch.LongTensor``(s) of shape ``(batch_size)``:
            Labels for computing classification losses. Requires ``mc_token_ids``.
            NOTE: The model has to be initialized with a config containing a dict ``cls`` with ``cl_labels_0``,
            ``cl_labels_1``, ... as keys where each entry holds the configuration for one classifier head with the
            mandatory entry ``labels``. Classifier heads are created as ``SequenceSummary`` module(s). The last entry of
            ``mc_token_ids`` (assumed to be the correct multiple choice entry) is used as index to calculate the
            summary from.
            NOTE: When providing ``is_adversarial: True`` as configuration entry for any cl head, gradients _below_ the
            cl head are inverted during training, i.e. the transformer model should become agnostic to the information
            provided by the corresponding cl labels.


    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
        **cl_loss_0, cl_loss_1,...**: (`optional`, returned when ``cl_labels_0``, ``cl_labels_1``, ... are provided) ``torch.FloatTensor``s of shape ``(1,)``:
            Classification loss(es).
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mc_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)``
            Prediction scores of the multiplechoice classification head (scores for each choice before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(2, batch_size, num_heads, sequence_length, embed_size_per_head)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(GPT2MultiHeadsAdversarialClModel, self).__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        if not hasattr(config, 'cls'):
            config.cls = {}

        # set default values
        if 'default' not in config.cls:
            config.cls['default'] = {
                "summary_first_dropout": 0.1,
                "summary_proj_to_labels": True,
                "summary_type": 'cls_index',
                "summary_use_proj": True,
                "is_adversarial": False,
            }

        self.cl_heads = {}
        for cl_name, cl_config in config.cls.items():
            if cl_name != 'default':
                assert 'labels' in cl_config, f'no labels set in config for classifier {cl_name}'
                _cl_config = copy.deepcopy(config.cls['default'])
                _cl_config.update(cl_config)
                _cl_config['num_labels'] = len(_cl_config['labels'])
                _cl_config['hidden_size'] = config.hidden_size
                self.cl_heads[cl_name] = SequenceSummary(PretrainedConfig(**_cl_config))
                setattr(self.cl_heads[cl_name], 'is_adversarial', _cl_config.get('is_adversarial', False))
                self.add_module(name=f'cl_head_{cl_name}', module=self.cl_heads[cl_name])

        config.num_labels = 1

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, mc_token_ids=None, lm_labels=None, mc_labels=None,
                **cl_labels_multi):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]
        # take only the last candidate hidden state and mc token id for classifier input
        hidden_states_cl = hidden_states[:, -1:]
        cl_token_ids = mc_token_ids[:, -1:]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        cl_logits = {}
        for input_name in cl_labels_multi.keys():
            assert input_name != 'default', f'"default" not allowed as kwarg (used as cl_labels_multi)'
            # reverse gradients, if head is adversarial (defaults to False)
            _hidden_states = grad_reverse(hidden_states) if self.cl_heads[input_name].is_adversarial else hidden_states
            cl_logits[input_name] = self.cl_heads[input_name](hidden_states_cl, cl_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + tuple(cl_logits.values()) + transformer_outputs[1:]
        losses = ()

        if len(cl_logits) > 0:
            loss_fct = CrossEntropyLoss()
            for input_name, logits in cl_logits.items():
                loss = loss_fct(logits.view(-1, logits.size(-1)),
                                cl_labels_multi[input_name].view(-1))
                losses = losses + (loss,)

        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                            mc_labels.view(-1))
            losses = (loss, ) + losses
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            losses = (loss, ) + losses

        return losses + outputs  # (lm loss), (mc loss), (cl_loss_0, cl_loss_1, ...), lm logits, mc logits, (cl_logits_0, cl_logits_1, ...), presents, (all hidden_states), (attentions)