import torch
import tensorflow as tf
import time
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

torch.cuda.empty_cache()

def load_dataset(file_path, tokenizer, block_size = 128):
    dataset = TextDataset(
        tokenizer = tokenizer,
        file_path = file_path,
        block_size = block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm = False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator

def train(train_file_path,model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
save_steps,save_total_limit):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "pad_token":"",
        "bos_token":"",
        "eos_token":""
    })
    tokenizer.add_tokens([":"])
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    model.save_pretrained(output_dir)
    print("Saved the pretrained model")

    training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
          save_total_limit=save_total_limit,
#           save_steps=save_steps
      )

    trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
    )
  trainer.train()
    trainer.save_model()
    print("Saved the trained model")


train_file_path = "/content/output.txt"
model_name = 'gpt2-medium'
output_dir = '/content/drive/MyDrive/chat_models'
overwrite_output_dir = True
per_device_train_batch_size = 2
num_train_epochs = 5
save_steps = 500
save_total_limit=2


total_start = time.time()

# Train
train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    save_total_limit=save_total_limit
)

print("######### TIME TAKEN FOR TOTAL CODE : {}".format(time.time() - total_start))



from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/chat_models')
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/chat_models')

# Set the device to use (e.g., 'cuda' for GPU or 'cpu' for CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(device)
# Function to generate text
def generate_text(prompt, max_length=500):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Set pad_token_id to eos_token_id for open-end generation
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(input_ids, max_length=max_length, attention_mask=attention_mask, pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Find the index of the first occurrence of ':' (colon)
    colon_index = generated_text.find(':')
    if colon_index != -1:
        # Check if there is another colon after the first one
        next_colon_index = generated_text.find(':', colon_index + 1)
        if next_colon_index != -1:
            # Insert '' only after the first colon
            generated_text = generated_text[:colon_index + 1] + '  ' + generated_text[colon_index + 1:]

    # Replace all other occurrences of '' with an empty string
    generated_text = generated_text.replace('', '')

    return generated_text.strip()  # Strip any leading or trailing whitespace
# Prompt for user input and generate text
while True:
    user_prompt = input("Enter a prompt (or 'exit' to quit): ")
    if user_prompt.lower() == 'exit':
        break
  exit_value = user_prompt
    user_prompt = " " + user_prompt + " : "

    # Generate text
    generated_output = generate_text(user_prompt)

    # Print only the generated output
    print("\nGenerated Text:")
    print(generated_output.split('\n')[0])  # Print only the first line of generated output

  from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/chat_models')
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/chat_models')

tokenizer.add_special_tokens({
    "pad_token": "",
    "bos_token": "",
    "eos_token": ""
})

tokenizer.add_tokens([":"])
model.resize_token_embeddings(len(tokenizer))

# Set the device to use (e.g., 'cuda' for GPU or 'cpu' for CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(device)

# Function to split input text into chunks
def split_into_chunks(text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

# Function to generate text for a single chunk
def generate_text_from_chunk(chunk, max_length=1024):
    input_ids = torch.tensor(chunk).unsqueeze(0).to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Set pad_token_id to eos_token_id for open-end generation
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(input_ids, max_length=max_length, attention_mask=attention_mask, pad_token_id=pad_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
return generated_text.strip()  # Strip any leading or trailing whitespace

# Function to generate text from the entire input
def generate_text(prompt, max_length=1024):
    chunks = split_into_chunks(prompt, max_length - 50)  # Leaving space for additional tokens and special tokens
    generated_output = ""

    for chunk in chunks:
        chunk_output = generate_text_from_chunk(chunk, max_length)
        generated_output += chunk_output + " "  # Ensure separation between chunks

    return generated_output

# Provide clear instructions in the prompt
def prepare_prompt(user_prompt):
    instruction = (
        "Convert the following Pascal code to Python code:\n"
        "Pascal:\n"
        f"{user_prompt}\n"
        "Python:\n"
    )
    return " " + instruction + " : "

# Prompt for user input and generate text
while True:
    user_prompt = input("Enter a prompt (or 'exit' to quit): ")
    if user_prompt.lower() == 'exit':
        break

    prompt_with_instruction = prepare_prompt(user_prompt)

    # Generate text
    generated_output = generate_text(prompt_with_instruction)

    # Print the generated output
    print("\nGenerated Text:")
    print(generated_output.strip())
