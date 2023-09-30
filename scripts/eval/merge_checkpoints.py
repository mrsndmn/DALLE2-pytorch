
import torch

configs = [
    '.decoder_full_no_prior_A100_u0/latest_checkpoint.pth',
    '.decoder_full_no_prior_A100_u1/latest_checkpoint.pth',
    '.decoder_full_no_prior_A100_u2/latest_checkpoint.pth',
]

u2_state = torch.load(configs[2], map_location='cpu')
u1_state = torch.load(configs[1], map_location='cpu')
u0_state = torch.load(configs[0], map_location='cpu')

u2_state['model'] = u2_state['model'].update(u1_state['model']).update(u0_state['model'])


