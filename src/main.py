import argparse
import json
import time
import config
import os
from llm import load_llm, load_vocab, load_base_ids
from files_content import load_user_input, load_func_def
from constraint_decoder import constraint_decoder
from json_handler import write_json

def main() -> None:
    os.environ["HF_HOME"] = "/home/asaph/42_school/Call-Me-Maybe/llm_sdk/.hf_cache"
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    # try:
    config.load_args(args=args)
    load_llm()
    start_time = time.time()
    load_vocab()
    load_user_input()
    load_func_def()
    load_base_ids()
    calls = json.loads(constraint_decoder())
    for c in calls:
        c["name"] = c["name"][4:]
    write_json(json.dumps(calls))

    print("Time taken: ", time.time() - start_time)
    # except Exception as e:
    #     print("An error occured: ", e)


if __name__ == "__main__":
    main()
