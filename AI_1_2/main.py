import requests
import yaml
import sys
import os
import json


HISTORY_FILE = os.path.join(os.path.dirname(__file__), "conversation_history.json")


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


class Agent:
    def __init__(self, api_config):
        self.api_config = api_config
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self._headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        self._load_history()
    
    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
                print(f"[Loaded {len(self.messages)} messages from history]")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Warning: Could not load history: {e}]")
    
    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[Warning: Could not save history: {e}]")
    
    def send_request(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        self._save_history()
        
        payload = {
            "model": self.api_config.get("model", "deepseek-chat"),
            "messages": self.messages,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.api_config["url"], 
                headers=self._headers, 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def receive_response(self, response):
        if response and 'choices' in response:
            assistant_reply = response['choices'][0]['message']['content']
            self.messages.append({"role": "assistant", "content": assistant_reply})
            self._save_history()
            return assistant_reply
        return None
    
    def process(self, user_message):
        response = self.send_request(user_message)
        return self.receive_response(response)
    
    def clear_history(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        print("History file deleted.")


def main():
    api_config = load_config()
    agent = Agent(api_config)
    
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
            agent.clear_history()
            print("Conversation cleared.")
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = agent.process(user_input)
        
        if response:
            print(response)
        else:
            print("Failed to get response. Try again.")


if __name__ == "__main__":
    main()
