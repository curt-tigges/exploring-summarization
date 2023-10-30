from transformer_lens.utils import get_act_name


def resid_names_filter(name: str):
    """Filter for the names of the activations we want to keep to study the resid stream."""
    return name.endswith("resid_post") or name == get_act_name("resid_pre", 0)
