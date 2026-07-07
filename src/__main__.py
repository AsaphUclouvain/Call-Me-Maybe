import argparse
from pathlib import Path
from json import loads, JSONDecodeError

FORMAT_RESPONSE ="""
{
    "name": "[function_name]",
    "parameters": [parameters]
}
"""

def get_str_prompts(file_path: str) -> list[str]:
    path = Path(file_path)
    try:
        text = path.read_text()
        dict_prompts = loads(text)
    except (OSError, JSONDecodeError)  as e:
        raise ValueError(f"impossible de charger les donnees depuis {file_path}") from e
    else:
        return [str(rup) for rup in dict_prompts]

def get_func_defs(file_path):
    path = Path(file_path)
    try:
        return path.read_text()
    except OSError as e:
        raise ValueError(f"impossible de charger les donnees depuis {file_path}") from e


def build_prompts(args: argparse.Namespace) -> list[str]:
    user_prompts = get_str_prompts(args.input)
    func_defs = get_func_defs(args.functions_definitions)
    prompts = []
    for up in user_prompts:
        prompt = "\n".join((up, func_defs, FORMAT_RESPONSE))
        prompts.append(prompt)
    return prompts
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    args = parser.parse_args()
    print(args)
    print("Hello from call-me-maybe!")


if __name__ == "__main__":
    main()
