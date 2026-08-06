import re
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Config
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME = "ShashiVish/cover-letter-dataset"
OUTPUT_DIR = "./llama2-7b-chat-coverletter-final"
MAX_LENGTH = 768
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 4
LEARNING_RATE = 2e-4
USE_CHAT_FORMAT = False
CLEAN_DATASET = True
SEED = 316
VAL_FRACTION = 0.1

# To login:
# huggingface-cli login

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# QLoRA config to reduce memory use
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load quantized base 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prep for k-bit training
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)

# Apply LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
raw_train_full = load_dataset(DATASET_NAME, split="train")
raw_test = load_dataset(DATASET_NAME, split="test")

# Split dataset (train/validation/test)
_split = raw_train_full.train_test_split(test_size=VAL_FRACTION, seed=SEED)
raw_train = _split["train"]
raw_val = _split["test"]

print(f"Splits — train: {len(raw_train)}, val: {len(raw_val)}, test: {len(raw_test)}")

# Remove trash 
def clean_cover_letter(text):
    if text is None:
        return ""

    parts = re.split(r"\n###\s*Sample", text)

    for p in parts:
        if "Dear" in p or len(p.strip()) > 100:
            cleaned = re.sub(r"#{2,}.*\n", "", p).strip()
            cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
            return cleaned

    cleaned = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return cleaned

# Prompt construction
def format_example(ex):
    system_prompt = (
        "You are an expert in professional communication. "
        "Write a polished, personalized cover letter based only on the following job description and resume."
    )

    base_prompt = (
        "### Job Description\n"
        f"Job Title: {ex.get('Job Title','')}\n"
        f"Company: {ex.get('Hiring Company','')}\n"
        f"Preferred Qualifications: {ex.get('Preferred Qualifications','')}\n\n"

        "### Applicant Resume\n"
        f"Name: {ex.get('Applicant Name','')}\n"
        f"Current Experience: {ex.get('Current Working Experience','')}\n"
        f"Past Experience: {ex.get('Past Working Experience','')}\n"
        f"Skills: {ex.get('Skillsets','')}\n"
        f"Qualifications: {ex.get('Qualifications','')}\n\n"

        "### Cover Letter\n"
        "Using the information above, write a professional, personalized cover letter."
    )

    completion_raw = ex.get("Cover Letter", "") or ""
    completion = clean_cover_letter(completion_raw) if CLEAN_DATASET else completion_raw.strip()

    # Optional: produce chat-style input
    if USE_CHAT_FORMAT:
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{base_prompt} [/INST] "
        return {"prompt": prompt, "completion": completion}

    prompt = base_prompt + "\n"
    return {"prompt": prompt, "completion": completion}


train_dataset = raw_train.map(format_example, remove_columns=raw_train.column_names)
val_dataset = raw_val.map(format_example, remove_columns=raw_val.column_names)
test_dataset = raw_test.map(format_example, remove_columns=raw_test.column_names)
 
# Drop empty completions
_pre = (len(train_dataset), len(val_dataset), len(test_dataset))
train_dataset = train_dataset.filter(lambda ex: len(ex["completion"].strip()) > 0)
val_dataset = val_dataset.filter(lambda ex: len(ex["completion"].strip()) > 0)
test_dataset = test_dataset.filter(lambda ex: len(ex["completion"].strip()) > 0)
print(f"Dropped empty completions — train: {_pre[0] - len(train_dataset)}, "
      f"val: {_pre[1] - len(val_dataset)}, test: {_pre[2] - len(test_dataset)}")

# Tokenization
def tokenize_fn(ex):
    prompt_ids = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]
 
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids.copy()
 
    truncated = len(input_ids) > MAX_LENGTH
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]
 
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "truncated": truncated,
    }
 
train_dataset = train_dataset.map(tokenize_fn, remove_columns=["prompt", "completion"])
val_dataset = val_dataset.map(tokenize_fn, remove_columns=["prompt", "completion"])
test_dataset = test_dataset.map(tokenize_fn, remove_columns=["prompt", "completion"])

# Report truncation for every split
for name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
    n_trunc = sum(ds["truncated"])
    print(f"Truncated at {MAX_LENGTH} tokens [{name}]: {n_trunc}/{len(ds)} "
          f"({100 * n_trunc / len(ds):.1f}%)")
 
# Drop `truncated` before training
train_dataset = train_dataset.remove_columns("truncated")
val_dataset = val_dataset.remove_columns("truncated")
test_dataset = test_dataset.remove_columns("truncated")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    report_to="none",
    gradient_checkpointing=True,
    seed=SEED,
    data_seed=SEED,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

 
print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model saved to:", OUTPUT_DIR)
 
with open(f"{OUTPUT_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
