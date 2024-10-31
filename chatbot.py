import nltk
from nltk.chat.util import Chat, reflections
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define pairs of patterns and responses for the rule-based chatbot
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today?",]
    ],
    [
        r"hi|hello|hey",
        ["Hello!", "Hi there!", "Hey!",]
    ],
    [
        r"how are you ?",
        ["I'm doing good. How about you?",]
    ],
    [
        r"what is your name ?",
        ["I'm a chatbot created for this example.",]
    ],
    [
        r"sorry (.*)",
        ["It's alright", "No worries!",]
    ],
    [
        r"i'm (.*) doing good",
        ["Glad to hear that!", "That's great to hear!"]
    ],
    [
        r"quit",
        ["Goodbye! Take care.",]
    ],
]

# Initialize the rule-based Chat instance
rule_based_chatbot = Chat(pairs, reflections)

# Initialize the AI-powered GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate AI-powered response
def generate_ai_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Choose mode of chatbot
mode = input("Choose chatbot mode - Type 'rule' for Rule-based or 'ai' for AI-powered: ").strip().lower()
print("Hello! I'm your chatbot. Type 'quit' to exit.")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    if mode == "rule":
        # Rule-based response
        response = rule_based_chatbot.respond(user_input)
        print(f"Chatbot: {response}")
    elif mode == "ai":
        # AI-powered response
        prompt = f"User: {user_input}\nChatbot:"
        response = generate_ai_response(prompt)
        print(f"Chatbot: {response}")
    else:
        print("Invalid mode selected.")
        break
