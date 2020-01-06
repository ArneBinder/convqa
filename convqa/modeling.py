from torch import nn
from torch.autograd import Function
from torch.nn import CrossEntropyLoss
from transformers import GPT2DoubleHeadsModel, GPT2PreTrainedModel, GPT2Model
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


#TODO: ADAPT FOR huggingface/transformers

class GPT2ClassifierHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, config, num_labels, invert_gradients=False):
        super(GPT2ClassifierHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
        self.num_labels = num_labels
        self.linear = nn.Linear(config.n_embd, self.num_labels)
        self.invert_gradients = invert_gradients


        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # Classification logits
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        # (bsz, num_choices, 1, hidden_size)
        classifier_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # invert gradients
        if self.invert_gradients:
            classifier_h = grad_reverse(classifier_h)
        # (bsz, num_choices, hidden_size)
        classifier_h = self.dropout(classifier_h.transpose(1, 2)).transpose(1, 2)
        classifier_logits = self.linear(classifier_h)
        # (bsz, num_choices, num_labels)
        return classifier_logits


class GPT2DoubleHeadsModelwithAdversarial(GPT2DoubleHeadsModel):
    """OpenAI GPT-2 model with a Language Modeling and a Multiple Choice head ("Language Models are Unsupervised Multitask Learners").

        Params:
            config: a GPT2Config class instance with the configuration to build a new model

        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
                indices selected in the range [0, config.vocab_size[
            `mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
                which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
            `position_ids`: an optional torch.LongTensor with the same shape as input_ids
                with the position indices (selected in the range [0, config.n_positions - 1[.
            `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
                You can use it to add a third type of embedding to each input token in the sequence
                (the previous two being the word and position embeddings).
                The input, position and token_type embeddings are summed inside the Transformer before the first
                self-attention block.
            `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
                with indices selected in [-1, 0, ..., config.vocab_size]. All labels set to -1 are ignored (masked), the loss
                is only computed for the labels set in [0, ..., config.vocab_size]
            `mc_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
                with indices selected in [0, ..., num_choices].
            `adv_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size, num_choices]
                with indices selected in [0, ..., num_adv_labels].

            `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states
                (key and values in the attention blocks) to speed up sequential decoding
                (this is the presents output of the model, cf. below).

        Outputs:
            if `lm_labels` and `multiple_choice_labels` are not `None`:
                Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
            else: a tuple with
                `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, config.vocab_size]
                `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
                `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as
                    torch.FloatTensors. They can be reused to speed up sequential decoding.

        Example usage:
        ```python
        # Already been converted into BPE token ids
        input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
        mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)

        config = modeling_gpt2.GPT2Config()

        model = modeling_gpt2.GPT2LMHeadModel(config)
        lm_logits, multiple_choice_logits, presents = model(input_ids, mc_token_ids)
        ```
        """
    def __init__(self, config, num_adv_labels, **kwargs):
        super(GPT2DoubleHeadsModelwithAdversarial, self).__init__(config, **kwargs)
        self.adv_classifier_head = GPT2ClassifierHead(config, num_labels=num_adv_labels, invert_gradients=True) if num_adv_labels > 1 else None
        self.num_adv_labels = num_adv_labels

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, adv_labels=None, token_type_ids=None,
                position_ids=None, past=None):
        transformer_output = self.transformer(input_ids, position_ids, token_type_ids, past)
        if self.transformer.output_attentions:
            all_attentions, hidden_states, presents = transformer_output
        else:
            hidden_states, presents = transformer_output
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        adv_logits = self.adv_classifier_head(hidden_states, mc_token_ids) if self.adv_classifier_head is not None else None
        losses = []
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if adv_labels is not None and adv_logits is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_adv = loss_fct(adv_logits.view(-1, self.adv_classifier_head.num_labels), adv_labels.view(-1))
            losses.append(loss_adv)
        if losses:
            return losses
        if self.transformer.output_attentions:
            return all_attentions, lm_logits, mc_logits, presents
        return lm_logits, mc_logits, presents


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

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
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

        if not hasattr(config, 'cl_labels'):
            config.cl_labels = {}
        if not hasattr(config, 'cl_is_adversarial'):
            config.cl_is_adversarial = {}
        self.cl_is_adversarial = config.cl_is_adversarial

        _num_labels = config.num_labels
        self.cl_heads = {}
        for input_name, cl_labels in config.cl_labels.items():
            config.num_labels = len(cl_labels)
            self.cl_heads[input_name] = SequenceSummary(config)
            self.add_module(name=f'cl_head_{input_name}', module=self.cl_heads[input_name])

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
            # reverse gradients, if head is adversarial (defaults to False)
            _hidden_states = grad_reverse(hidden_states) if self.cl_is_adversarial.get(input_name, False) else hidden_states
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