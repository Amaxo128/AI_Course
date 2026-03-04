import requests
import yaml
import sys
import os


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


def send_message(messages, api_config):
    headers = {
        "Authorization": f"Bearer {api_config['key']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": api_config.get("model", "deepseek-chat"),
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


def get_mode():
    print("\nSelect answer mode:")
    print("1 - Just get answer")
    print("2 - Solve step by step")
    print("3 - Generate prompt then answer")
    print("4 - Multiple experts (analyst, manager, critic)")
    
    while True:
        try:
            choice = input("\nChoice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            return 1


def transform_message(user_input, mode):
    if mode == 1:
        return user_input
    elif mode == 2:
        return f"{user_input}\n\nPlease solve this step by step, showing your reasoning at each step."
    elif mode == 4:
        return f"""You are a panel of three experts: an Analyst, a Manager, and a Critic.

For the following question, each expert should provide their perspective:

User question: {user_input}

Please have each expert answer in their style:
- Analyst: Focus on data, facts, and analytical reasoning
- Manager: Focus on practical implementation, resources, and outcomes
- Critic: Focus on potential issues, risks, and weaknesses

Provide all three perspectives."""
    return user_input


def main():
    api_config = load_config()
    
    print("DeepSync Terminal Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("Type 'mode' to change answer mode")
    print("-" * 40)
    
    current_mode = get_mode()
    mode_names = {1: "Just answer", 2: "Step by step", 3: "Generate prompt then answer", 4: "Multiple experts"}
    print(f"Current mode: {current_mode} - {mode_names.get(current_mode, 'Unknown')}")
    
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
        
        if user_input.lower() == 'mode':
            current_mode = get_mode()
            mode_names = {1: "Just answer", 2: "Step by step", 3: "Generate prompt then answer", 4: "Multiple experts"}
            print(f"Mode changed to: {current_mode} - {mode_names.get(current_mode, 'Unknown')}")
            continue
        
        if current_mode == 3:
            print("\n[Generating prompt...]")
            prompt_gen_messages = [
                {"role": "system", "content": "You are a prompt enhancer. Generate an improved, detailed prompt from the user's question that will help get a better answer. Only output the enhanced prompt, nothing else."},
                {"role": "user", "content": user_input}
            ]
            prompt_response = send_message(prompt_gen_messages, api_config)
            
            if prompt_response and 'choices' in prompt_response:
                generated_prompt = prompt_response['choices'][0]['message']['content'].strip()
                print(f"\n[Generated prompt]\n{generated_prompt}\n")
                print("[Getting answer...]")
                
                messages.append({"role": "user", "content": generated_prompt})
            else:
                print("Failed to generate prompt. Trying direct answer...")
                messages.append({"role": "user", "content": user_input})
        else:
            transformed_input = transform_message(user_input, current_mode)
            messages.append({"role": "user", "content": transformed_input})
        
        print("\nAssistant: ", end="", flush=True)
        response = send_message(messages, api_config)
        
        if response and 'choices' in response:
            assistant_reply = response['choices'][0]['message']['content']
            print(assistant_reply)
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            print("Failed to get response. Try again.")


if __name__ == "__main__":
    main()
