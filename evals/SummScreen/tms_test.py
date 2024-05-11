# predictions, answers, lengths = [], [], []
# Copyright (c) 2023, Tri Dao, Albert Gu.
# TOKENIZERS_PARALLELISM=false python ./evals/multiqa_pred.py --dataset_dir ./dataset/multiqa/hotpotqa.jsonl --model-name "EleutherAI/pythia-1.4b" --minp 0.05 --topk 1 --temperature 0.7 --repetition-penalty 1.2 --out_dir ./outputs
# TOKENIZERS_PARALLELISM=false python ./evals/multiqa_pred.py --dataset_dir ./dataset/multiqa/hotpotqa.jsonl --model-name state-spaces/mamba-1.4b --minp 0.05 --topk 1 --temperature 0.7 --repetition-penalty 1.2 --out_dir ./outputs
import argparse
import time
import json
import os
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import train_helper

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--dataset_dir", type=str, default="./dataset/SummScreen/TVMegaSite/tms_test.jsonl")
parser.add_argument("--model_path", type=str, default="state-spaces/mamba-1.4b")
parser.add_argument("--state_path", type=str, default=None)
parser.add_argument("--model_name", type=str, default="mamba-1.4b")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=50)
parser.add_argument("--genlen", type=int, default=600)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--out_dir", type=str, default="./output")
args = parser.parse_args()

device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_path}")
is_mamba = args.model_name.startswith("mamba")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_path, device=device, dtype=dtype)
    if args.state_path is not None:
        model.load_state_dict(torch.load(args.state_path))
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device="cuda")
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
with open(args.dataset_dir, "r") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [json.loads(line) for line in lines]
dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "test_with_sample": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation. Here is an example: "
            "\n\n Story: Once upon a time, in a small coastal town, there was a girl named Lily. Lily had always been fascinated by the sea and its mysteries. Every day after school, she would rush to the shore to collect seashells and watch the waves crash against the rocks.\nOne stormy afternoon, as dark clouds loomed overhead and the wind howled, Lily spotted something glimmering among the rocks. Ignoring the pelting rain, she hurried closer and discovered a chest half-buried in the sand. With trembling hands, she pried it open to reveal a map adorned with strange markings and cryptic symbols."
            "\nDetermined to uncover the map's secrets, Lily embarked on a thrilling adventure. She deciphered the clues, braved treacherous cliffs and deep caves, and faced countless challenges along the way. With each obstacle overcome, Lily grew more determined to unlock the treasure hidden at the map's end."
            "\nfter days of searching, Lily finally reached the spot marked on the map—a secluded cove where the sun cast golden rays upon the sparkling sand. Digging eagerly, she unearthed a chest filled with precious gems and ancient artifacts, a testament to her courage and perseverance."
            "\nAs she gazed at the treasure shimmering in the sunlight, Lily realized that the greatest reward of all was not the riches she had found, but the unforgettable journey that had led her here."
            "\n\nQuestion: What qualities does Lily possess that enable her to embark on a perilous quest and uncover hidden treasures?"
            "\n\nAnswer: Lily possesses qualities of curiosity, determination, courage, and perseverance, which enable her to embark on a perilous quest and uncover hidden treasures."
            "\n\nHere is your story and question:"
            "\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion:{input}\n\nAnswer:",
    "test": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation."
            "\n\nHere is your story and question:"
            "\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion:{input}\n\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "tms_test": "Summarize the content into a few short sentences. Content:{context}\n\nSummary:\n"
}
dataset = args.dataset_dir.split("/")[-1].split(".")[0]
dataset_rename = re.sub(r"_e(?!n)", "", dataset)
if not os.path.exists(args.out_dir+"/{}".format(args.model_name)):
    os.makedirs(args.out_dir+"/{}".format(args.model_name))
file_name = args.out_dir+"/{}/{}.txt".format(args.model_name,dataset)
ref_file_name = args.out_dir+"/{}/ref-{}.txt".format(args.model_name,dataset)
with open(ref_file_name, "w") as ref_fp:
    for answer in lines:
        ref_fp.write(answer["response"] + "\n")
start = time.time()
answers = []
gen_length = []
for i, line in enumerate(tqdm(lines)):
    prompt = dataset2prompt.get(dataset_rename, None)
    prompt = prompt.format(context=line["context"])
    if prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(prompt, return_tensors="pt")
        # 把prompt中前i个位置的tokens
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + args.genlen
    model.eval()
    with torch.no_grad():
        if is_mamba:
            fn = lambda: model.generate(
                input_ids=input_ids,
                max_length=max_length,
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                min_p=args.minp,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            fn = lambda: model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=max_length,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                repetition_penalty=args.repetition_penalty,
            )
        out = fn()
        if prompt is not None:
            # print(tokenizer.batch_decode(out.sequences.tolist()))
            # clear '|endoftext|' tokens while decoding
            answer = tokenizer.decode(out.sequences[0].tolist()[len(input_ids[0]):])
            answer = answer.split("<|endoftext|>")[0]
            print(answer)

            answers.append(answer)
            gen_length.append(len(answer))
            # lines[i]["pred"] = answer


torch.cuda.synchronize()
print(f"{args.model_name} processing + time: {(time.time() - start)*1000:.0f}ms")
with open(file_name, "w") as fp:
    # for answer in answers:
    #     fp.write(answer + "\n")
    bleu_score = train_helper.run_multi_bleu(file_name, ref_file_name)
    print(bleu_score)
    fp.write("avg generation steps: {}, bleu: {:.4f}".format(
    sum(gen_length) / len(gen_length), bleu_score))


