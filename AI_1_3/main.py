import requests
import yaml
import sys
import os

try:
    import tiktoken
except ImportError:
    tiktoken = None


def count_tokens(text, encoder=None):
    if encoder:
        return len(encoder.encode(text))
    return len(text) // 4


def count_messages_tokens(messages, encoder=None):
    total = 0
    for msg in messages:
        total += count_tokens(msg["content"], encoder)
        total += 4
    total += 2
    return total


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
        self.encoder = None
        self.total_tokens_used = 0
        if tiktoken:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    
    def count_current_request_tokens(self):
        if self.messages:
            user_msg = self.messages[-1] if self.messages[-1]["role"] == "user" else None
            if user_msg:
                return count_tokens(user_msg["content"], self.encoder)
        return 0
    
    def count_history_tokens(self):
        return count_messages_tokens(self.messages, self.encoder)
    
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
            if response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                if 'maximum context length' in error_msg.lower() or 'too many tokens' in error_msg.lower():
                    print("\n⚠️  ERROR: Maximum context length exceeded!")
                    print(f"   Message: {error_msg}")
                    self.messages.pop()
                    return None
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
            
            usage = response.get('usage', {})
            api_prompt_tokens = usage.get('prompt_tokens', 0)
            api_completion_tokens = usage.get('completion_tokens', 0)
            api_total_tokens = usage.get('total_tokens', 0)
            
            self.total_tokens_used += api_total_tokens
            
            local_request_tokens = self.count_current_request_tokens()
            local_history_tokens = self.count_history_tokens()
            local_response_tokens = count_tokens(assistant_reply, self.encoder)
            
            print("\n" + "=" * 40)
            print("📊 Token Usage:")
            print(f"  Current request:   {local_request_tokens} tokens")
            print(f"  Response:          {local_response_tokens} tokens")
            print(f"  History total:    {local_history_tokens} tokens")
            print("-" * 40)
            print("📡 API Usage:")
            print(f"  Prompt tokens:     {api_prompt_tokens}")
            print(f"  Completion tokens: {api_completion_tokens}")
            print(f"  Total tokens:      {api_total_tokens}")
            print(f"  Session total:     {self.total_tokens_used}")
            print("=" * 40)
            
            return assistant_reply
        return None
    
    def process(self, user_message):
        response = self.send_request(user_message)
        return self.receive_response(response)
    
    def clear_history(self):
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.total_tokens_used = 0


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
