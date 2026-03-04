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


def send_message(messages, api_config, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {api_config['key']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": api_config.get("model", "deepseek-chat"),
        "messages": messages,
        "temperature": temperature,
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


def send_multi_temperature(messages, api_config):
    temperatures = [0, 0.7, 1.2]
    responses = []
    
    for temp in temperatures:
        temp_messages = [msg for msg in messages]
        response = send_message(temp_messages, api_config, temperature=temp)
        if response and 'choices' in response:
            responses.append({
                'temperature': temp,
                'content': response['choices'][0]['message']['content']
            })
        else:
            responses.append({
                'temperature': temp,
                'content': None
            })
    
    return responses


def main():
    api_config = load_config()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    print("DeepSync Terminal Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
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
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            print("Conversation cleared.")
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        print("\nGetting responses with different temperatures...")
        responses = send_multi_temperature(messages, api_config)
        
        for resp in responses:
            print(f"\n--- Temperature {resp['temperature']} ---")
            if resp['content']:
                print(resp['content'])
                messages.append({"role": "assistant", "content": resp['content']})
            else:
                print("Failed to get response.")


if __name__ == "__main__":
    main()
