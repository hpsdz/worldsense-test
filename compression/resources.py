"""Machine-local auxiliary resources, separate from experimental parameters."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(audioclip='resources/AudioCLIP-Full-Training.pt',
                clip_model='resources/clip-vit-large-patch14-336', nltk_data='assets/nltk_data')

def resource_path(name, override=None):
    config = ROOT / 'resources.local.json'
    settings = json.loads(config.read_text(encoding='utf-8-sig')) if config.exists() else {}
    if not isinstance(settings, dict) or set(settings)-set(DEFAULTS):
        raise ValueError('resources.local.json allows only audioclip, clip_model, nltk_data')
    value = override if override is not None else os.environ.get('QWEN_'+name.upper(), settings.get(name, DEFAULTS[name]))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Resource {name} must be a nonempty path string')
    return str((ROOT / Path(os.path.expandvars(value)).expanduser()).resolve())

def check_resources(method):
    names = {'omniselect': ['audioclip','nltk_data'], 'omniscope': ['clip_model']}[method]
    missing = []
    for name in names:
        path = Path(resource_path(name))
        if name == 'clip_model':
            valid = all((path/f).is_file() for f in ('config.json','preprocessor_config.json','vocab.json','merges.txt'))
            valid = valid and any((path/f).is_file() for f in ('model.safetensors','pytorch_model.bin'))
        elif name == 'nltk_data':
            valid = all((path/f).is_dir() for f in ('tokenizers/punkt_tab/english','taggers/averaged_perceptron_tagger_eng'))
        else:
            valid = path.is_file()
        print(f'{name}: {"OK" if valid else "MISSING"} {path}')
        if not valid: missing.append(name)
    return missing

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check auxiliary resources without loading model weights')
    parser.add_argument('--method', choices=['omniselect','omniscope'], required=True)
    raise SystemExit(1 if check_resources(parser.parse_args().method) else 0)
