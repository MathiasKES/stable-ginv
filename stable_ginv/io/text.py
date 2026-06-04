"""CLI argument-string parsing helpers."""


def parse_prefixes_with_fracs(prefixes_str):
    """Parse 'conv1:0.5,layer1:1.0,fc' into (prefixes_tuple, fracs_dict)."""
    prefixes = []
    fracs = {}
    for item in prefixes_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            prefix, frac = item.split(":", 1)
            prefix = prefix.strip()
            fracs[prefix] = float(frac.strip())
            prefixes.append(prefix)
        else:
            prefixes.append(item)
    return tuple(prefixes), fracs
