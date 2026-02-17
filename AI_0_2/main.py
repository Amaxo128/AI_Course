import requests
import yaml
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


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
    request_messages = list(messages)
    if api_config.get("response_format") == "json_object":
        combined_content = " ".join(
            str(msg.get("content", "")) for msg in request_messages if isinstance(msg, dict)
        ).lower()
        if "json" not in combined_content:
            request_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Respond in valid JSON format."
                }
            )
    else:
        request_messages = [
            msg for msg in request_messages
            if not (msg.get("role") == "system" and msg.get("content") == "Respond in valid JSON format.")
        ]

    headers = {
        "Authorization": f"Bearer {api_config['key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openrouter/auto-api",
        "X-Title": "DeepSync"
    }
    
    payload = {
        "model": api_config.get("model", "deepseek-chat"),
        "messages": request_messages,
        "stream": False
    }
    
    if api_config.get("max_tokens"):
        payload["max_tokens"] = api_config["max_tokens"]
    
    if api_config.get("stop_sequence"):
        payload["stop"] = api_config["stop_sequence"]
    
    if api_config.get("response_format"):
        if api_config["response_format"] == "json_object":
            payload["response_format"] = {"type": "json_object"}
        else:
            payload["response_format"] = {"type": "text"}
    
    try:
        response = requests.post(api_config["url"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def main():
    api_config = load_config()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    print("DeepSync Terminal Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("Type 'settings' to view/toggle LLM settings")
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
        
        if user_input.lower() == 'settings':
            current_response_format = api_config.get('response_format') or 'text'
            print("\n--- LLM Settings ---")
            print(f"max_tokens: {api_config.get('max_tokens', 'unlimited')}")
            print(f"stop_sequence: {api_config.get('stop_sequence', 'none')}")
            print(f"response_format: {current_response_format}")
            print("\nCommands to change settings:")
            print("  set max_tokens <number>")
            print("  set stop_sequence <sequence>")
            print("  set stop_sequence none - to remove stop sequence")
            print("  set response_format <text|json_object>")
            print("  set response_format none - same as text (default)")
            continue
        
        if user_input.lower().startswith('set '):
            parts = user_input[4:].split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                if key == 'max_tokens':
                    try:
                        api_config['max_tokens'] = int(value)
                        print(f"max_tokens set to: {value}")
                    except ValueError:
                        print("Error: max_tokens must be a number")
                elif key == 'stop_sequence':
                    if value.lower() == 'none':
                        api_config['stop_sequence'] = None
                        print("stop sequence removed")
                    else:
                        api_config['stop_sequence'] = value
                        print(f"stop sequence set to: {value}")
                elif key == 'response_format':
                    value_lower = value.lower()
                    if value_lower == 'none':
                        api_config['response_format'] = None
                        print("response format set to default text")
                    elif value_lower in ['text', 'json_object']:
                        if value_lower == 'text':
                            api_config['response_format'] = None
                            print("response format set to default text")
                        else:
                            api_config['response_format'] = value_lower
                            print(f"response format set to: {value_lower}")
                    else:
                        print("Error: response_format must be 'text', 'json_object', or 'none'")
                else:
                    print(f"Unknown setting: {key}")
            else:
                print("Usage: set <response_format|max_tokens|stop_sequence> <value>")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSync Terminal Chat")
    parser.add_argument("message", nargs="?", help="Message to send (for non-interactive mode)")
    parser.add_argument("--no-settings", action="store_true", help="Run without extra LLM settings")
    args = parser.parse_args()
    
    if args.message:
        api_config = load_config()
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        request_messages = messages + [{"role": "user", "content": args.message}]
        if api_config.get("response_format") == "json_object":
            combined_content = " ".join(
                str(msg.get("content", "")) for msg in request_messages if isinstance(msg, dict)
            ).lower()
            if "json" not in combined_content:
                request_messages.insert(
                    0,
                    {"role": "system", "content": "Respond in valid JSON format."}
                )
        else:
            request_messages = [
                msg for msg in request_messages
                if not (msg.get("role") == "system" and msg.get("content") == "Respond in valid JSON format.")
            ]
        
        if args.no_settings:
            payload = {
                "model": api_config.get("model", "deepseek-chat"),
                "messages": request_messages,
                "stream": False
            }
        else:
            payload = {
                "model": api_config.get("model", "deepseek-chat"),
                "messages": request_messages,
                "stream": False
            }
            if api_config.get("max_tokens"):
                payload["max_tokens"] = api_config["max_tokens"]
            if api_config.get("stop_sequence"):
                payload["stop"] = api_config["stop_sequence"]
            if api_config.get("response_format"):
                if api_config["response_format"] == "json_object":
                    payload["response_format"] = {"type": "json_object"}
                else:
                    payload["response_format"] = {"type": "text"}
        
        headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openrouter/auto-api",
            "X-Title": "DeepSync"
        }
        
        try:
            response = requests.post(api_config["url"], headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'choices' in result:
                print(result['choices'][0]['message']['content'])
            else:
                print(result)
        except Exception as e:
            print(f"Error: {e}")
    else:
        main()
