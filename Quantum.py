import torch
from torch import nn


class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase


        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)

        # self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))

    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)
        # batch_pos_embed = pos_embed.unsqueeze(dim = 0).expand_as(x)

        return pos_embed


class ComplexMultiply(torch.nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]

        if amplitude.dim() == phase.dim() + 1:  # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)

        elif amplitude.dim() == phase.dim():  # Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)


        else:
            raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        real_part = cos * amplitude
        imag_part = sin * amplitude

        return [real_part, imag_part]


class QOuter(torch.nn.Module):
    def __init__(self):
        super(QOuter, self).__init__()

    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(x)) + ' inputs.')

        # x[0], x[1] has shape:
        # (batch_size, time_stamps, embedding_dim)
        real = x[0].transpose(0, 1)
        imag = x[1].transpose(0, 1)
        output = []
        for r, i in zip(real, imag):
            output_rr = []
            output_ii = []
            for rr, ii in zip(r, i):
                unsqueezed_rr = torch.unsqueeze(rr, dim=-1)
                unsqueezed_ii = torch.unsqueeze(ii, dim=-1)
                _r = torch.mm(unsqueezed_rr, unsqueezed_rr.t()) + torch.mm(unsqueezed_ii, unsqueezed_ii.t())
                _i = -torch.mm(unsqueezed_rr, unsqueezed_ii.t()) + torch.mm(unsqueezed_ii, unsqueezed_rr.t())

                output_rr.append(_r)
                output_ii.append(_i)

            output_rr = torch.stack(output_rr, dim=0)
            output_ii = torch.stack(output_ii, dim=0)
            output.append([output_rr, output_ii])

        return output


class QMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, device=torch.device('cuda')):
        super(QMeasurement, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.kernel = torch.nn.Parameter(
            torch.stack([torch.eye(embed_dim).to(self.device), torch.zeros(embed_dim, embed_dim).to(self.device)],
                        dim=-1))

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        input_real = inputs[0]
        input_imag = inputs[1]

        real_kernel = self.kernel[:, :, 0]
        imag_kernel = self.kernel[:, :, 1]

        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
                         + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
                         - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        output_real = torch.matmul(torch.flatten(input_real, start_dim=-2, end_dim=-1),
                                   torch.flatten(projector_real, start_dim=-2, end_dim=-1).t()) \
                      - torch.matmul(torch.flatten(input_imag, start_dim=-2, end_dim=-1),
                                     torch.flatten(projector_imag, start_dim=-2, end_dim=-1).t())

        return output_real



class QMixture(torch.nn.Module):

    def __init__(self, use_weights=True, device=torch.device('cuda')):
        super(QMixture, self).__init__()
        self.use_weights = use_weights
        self.device = device

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        in_modalities = inputs[0]  # [modal_1,...modal_n], each being a list of [real, imag] arrays

        weights = inputs[1].transpose(0, 1)  # (time_stamps, batch_size, num_modalities)
        embed_dim = in_modalities[0][0][0].shape[-1]
        outputs = []
        for reps_t in zip(*in_modalities, weights):
            multimodal_rep = [torch.stack(rep_field, dim=-1) for rep_field in zip(*reps_t[:-1])]
            w = reps_t[-1].unsqueeze(dim=1).unsqueeze(dim=-1).expand(-1, embed_dim, -1, -1)
            output_rep = [torch.matmul(_rep, w).squeeze(dim=-1) for _rep in multimodal_rep]
            outputs.append(output_rep)

        return outputs


class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase

        # Vaswani et al.
        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        self.frequency_embedding.weight.requires_grad = True
        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)
        self.phase_embedding.weight.requires_grad = True
        # self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))

    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)
        # batch_pos_embed = pos_embed.unsqueeze(dim = 0).expand_as(x)

        return pos_embed
