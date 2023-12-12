import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers import PreTrainedModel, RobertaModel, RobertaConfig

import torch.nn.functional as F
from torch import nn
from Quantum import PositionEmbedding, ComplexMultiply, QOuter, QMixture, QMeasurement


class L2Norm(torch.nn.Module):

    def __init__(self, dim=1, keep_dims=True, eps = 1e-10):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims
        self.eps = eps

    def forward(self, inputs):

        output = torch.sqrt(self.eps+ torch.sum(inputs**2, dim=self.dim, keepdim=self.keepdim))

        return output




class BertQPENTagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        bert_config: configuration for bert model
        """
        super(BertQPENTagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels

        # initialized with pre-trained BERT and perform finetuning
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size

        ## quantum modules
        self.seq_len = 200
        self.dim = 50
        self.emb_dim = 100
        self.liner = nn.Linear(self.seq_len, self.dim)
        self.norm = L2Norm(dim=-1)
        self.projections = nn.Linear(penultimate_hidden_size, self.emb_dim)
        self.phase_embeddings = PositionEmbedding(self.emb_dim, input_dim=1)
        self.multiply = ComplexMultiply()
        self.mixture = QMixture()
        self.outer = QOuter()
        self.measurement = QMeasurement(self.emb_dim)

        # classifier
        self.classifier = nn.Linear(penultimate_hidden_size + self.dim + self.emb_dim, bert_config.num_labels)





    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        ## add
        utterance_reps = nn.ReLU()(self.projections(tagger_input))

        phases = self.phase_embeddings(attention_mask)
        amplitudes = F.normalize(utterance_reps, dim = -1)

        unimodal_pure = self.multiply([phases, amplitudes])
        unimodal_matrices = self.outer(unimodal_pure)


        weights = self.norm(utterance_reps)
        weights = F.softmax(weights, dim=-1)

        in_states = self.mixture([[unimodal_matrices], weights])

        output = []
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            #_output = self.fc_out(measurement_probs)
            output.append(measurement_probs)

        output = torch.stack(output, dim=-2)
        # print('output.shape:', output.shape)

        tagger_input = tagger_input @ torch.transpose(tagger_input, -1, -2)
        tagger_input = nn.ReLU()(self.liner(tagger_input))
        tagger_input = torch.cat([outputs[0], tagger_input, output], dim=-1)
        ##----------------------------------------------------------------------


        tagger_input = self.bert_dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)


        logits = self.classifier(tagger_input)
        #print('logits.shape:', logits.shape)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        return outputs


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple 
    interface for downloading and loading pretrained models.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class XLMRQPENTagger(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        ## quantum modules
        self.seq_len = 200
        self.dim = 50
        self.emb_dim = 100
        self.liner = nn.Linear(self.seq_len, self.dim)
        self.norm = L2Norm(dim=-1)
        self.projections = nn.Linear(config.hidden_size, self.emb_dim)
        self.phase_embeddings = PositionEmbedding(self.emb_dim, input_dim=1)
        self.multiply = ComplexMultiply()
        self.mixture = QMixture()
        self.outer = QOuter()
        self.measurement = QMeasurement(self.emb_dim)


        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + self.dim + self.emb_dim, config.num_labels)
        self.init_weights()





    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]

        ## add
        utterance_reps = nn.ReLU()(self.projections(tagger_input))

        phases = self.phase_embeddings(attention_mask)
        amplitudes = F.normalize(utterance_reps, dim=-1)

        unimodal_pure = self.multiply([phases, amplitudes])
        unimodal_matrices = self.outer(unimodal_pure)
        ## unimodal_matrices.shape = 200 * 2 * 16 * 50 * 50

        weights = self.norm(utterance_reps)
        weights = F.softmax(weights, dim=-1)
        ## weights.shape = 16 * 200 * 1

        in_states = self.mixture([[unimodal_matrices], weights])

        output = []
        for _h in in_states:
            measurement_probs = self.measurement(_h)
            # _output = self.fc_out(measurement_probs)
            output.append(measurement_probs)

        output = torch.stack(output, dim=-2)


        tagger_input = tagger_input @ torch.transpose(tagger_input, -1, -2)
        tagger_input = nn.ReLU()(self.liner(tagger_input))
        #tagger_input = self.dropout(tagger_input)
        tagger_input = torch.cat([outputs[0], tagger_input, output], dim=-1)


        tagger_input = self.dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)


        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        return outputs
