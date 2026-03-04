import requests
import yaml
import sys
import os


DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
]


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print("Error: config.yaml not found. Please create it with your API credentials.")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    api_config = config.get("api", {})
    
    if not api_config.get("key") or api_config.get("key") == "your_api_key_here":
        print("Error: Please set your API key in config.yaml")
        sys.exit(1)
    
    return api_config


def send_message(messages, api_config, model=None):
    headers = {
        "Authorization": f"Bearer {api_config['key']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model or api_config.get("model", "deepseek-chat"),
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def run_all_models(question, api_config):
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    
    print(f"\n{'='*50}")
    print(f"Running question across all DeepSeek models:")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    for model in DEEPSEEK_MODELS:
        print(f"\n--- {model} ---")
        messages = [system_msg, {"role": "user", "content": question}]
        
        response = send_message(messages, api_config, model=model)
        
        if response and 'choices' in response:
            assistant_reply = response['choices'][0]['message']['content']
            print(f"Response: {assistant_reply}")
        else:
            print(f"Failed to get response from {model}")
    
    print(f"\n{'='*50}")
    print("All models completed!")
    print(f"{'='*50}\n")


def run_single_model(question, api_config):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            print("Conversation cleared.")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="", flush=True)
        response = send_message(messages, api_config)
        
        if response and 'choices' in response:
            assistant_reply = response['choices'][0]['message']['content']
            print(assistant_reply)
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            print("Failed to get response. Try again.")


def main():
    api_config = load_config()
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_all_models(question, api_config)
        return
    
    print("DeepSync Terminal Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("Type 'all' to run question across all models")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            print("Conversation cleared.")
            continue
        
        if user_input.lower() == 'all':
            print("Enter question to run across all models:")
            question = input("Question: ").strip()
            if question:
                run_all_models(question, api_config)
            continue
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        
        print("\nAssistant: ", end="", flush=True)
        response = send_message(messages, api_config)
        
        if response and 'choices' in response:
            assistant_reply = response['choices'][0]['message']['content']
            print(assistant_reply)
        else:
            print("Failed to get response. Try again.")


if __name__ == "__main__":
    main()
