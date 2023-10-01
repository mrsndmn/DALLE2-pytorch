
import torch

configs = [
    '.decoder_full_no_prior_A100_u0/latest_checkpoint.pth',
    '.decoder_full_no_prior_A100_u1_backup/latest_checkpoint.pth',
    '.decoder_full_no_prior_A100_u2/latest_checkpoint.pth',
]

u2_state = torch.load(configs[2], map_location='cpu')
u2_state_keys = len(u2_state['model'].keys())

u1_state = torch.load(configs[1], map_location='cpu')
u1_values = {}
for k, v in u1_state['model'].items():
    if not k.startswith('noise_schedulers.1.') and not k.startswith('unets.1.'):
        continue
    # if k in u2_state['model']:
    #     print(k, "already in u2_state")
        continue

    u1_values[k] = v

u1_state_keys = len(u1_values.keys())

u0_state = torch.load(configs[0], map_location='cpu')
u0_values = {}
for k, v in u0_state['model'].items():
    if not k.startswith('noise_schedulers.0.') and not k.startswith('unets.0.'):
        continue
    # if k in u2_state['model']:
    #     print(k, "already in u2_state")
        continue

    u0_values[k] = v

u0_state_keys = len(u0_values.keys())

u2_state['model'].update(u1_values)
u2_state['model'].update(u0_values)

# expected_keys = u2_state_keys + u1_state_keys + u0_state_keys
# assert expected_keys == len(u2_state['model'].keys()), f"{expected_keys} == {len(u2_state['model'].keys())}"

torch.save(u2_state, '.decoder_full_no_prior_A100_full/latest_checkpoint.pth')
