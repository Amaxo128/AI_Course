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


class Agent:
    def __init__(self, api_config):
        self.api_config = api_config
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self._headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
    
    def send_request(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        
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
            return assistant_reply
        return None
    
    def process(self, user_message):
        response = self.send_request(user_message)
        return self.receive_response(response)
    
    def clear_history(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]


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
