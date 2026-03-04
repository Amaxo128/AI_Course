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


class ContextManager:
    def __init__(self, keep_recent=10, compress_every=10):
        self.keep_recent = keep_recent
        self.compress_every = compress_every
        self.summary = ""
        self.messages = []
    
    def build_messages(self, system_prompt):
        result = [{"role": "system", "content": system_prompt}]
        if self.summary:
            result.append({"role": "system", "content": f"Previous conversation summary:\n{self.summary}"})
        result.extend(self.messages)
        return result
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def should_compress(self):
        return len(self.messages) >= self.compress_every + self.keep_recent
    
    def get_old_messages(self):
        if len(self.messages) <= self.keep_recent:
            return []
        return self.messages[:-self.keep_recent]
    
    def compress(self, summary):
        old_count = len(self.messages)
        self.summary = summary
        self.messages = self.messages[-self.keep_recent:]
        return old_count


class Agent:
    def __init__(self, api_config, use_compression=False, keep_recent=10, compress_every=10):
        self.api_config = api_config
        self.system_prompt = "You are a helpful assistant."
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._headers = {
            "Authorization": f"Bearer {api_config['key']}",
            "Content-Type": "application/json"
        }
        self.encoder = None
        self.total_tokens_used = 0
        self.use_compression = use_compression
        self.context_manager = ContextManager(keep_recent, compress_every) if use_compression else None
        self.compression_count = 0
        self.tokens_saved = 0
        if tiktoken:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    
    def generate_summary(self, old_messages):
        summary_prompt = (
            "Создай краткое резюме (summary) этого разговора. "
            "Включи только ключевые темы, решения и факты. "
            "Используй не более 200 слов.\n\n"
            "Разговор:\n"
        )
        for msg in old_messages:
            summary_prompt += f"{msg['role']}: {msg['content']}\n"
        
        payload = {
            "model": self.api_config.get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": "Ты - ассистент, который создаёт краткие резюме разговоров."},
                {"role": "user", "content": summary_prompt}
            ],
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
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Summary generation error: {e}")
            return "Разговор был, но резюме не удалось создать."
    
    def check_and_compress(self):
        if not self.use_compression or not self.context_manager:
            return
        
        if self.context_manager.should_compress():
            print(f"\n🔄 Сжатие контекста (сообщений: {len(self.context_manager.messages)})...")
            
            old_messages = self.context_manager.get_old_messages()
            old_tokens = count_messages_tokens(old_messages, self.encoder)
            
            summary = self.generate_summary(old_messages)
            self.context_manager.compress(summary)
            
            summary_tokens = count_tokens(summary, self.encoder)
            saved = old_tokens - summary_tokens
            self.tokens_saved += max(0, saved)
            self.compression_count += 1
            
            print(f"   Сжато {len(old_messages)} сообщений → summary ({saved} токенов экономии)")
            print(f"\n   📝 Summary сохранено:\n{summary[:300]}{'...' if len(summary) > 300 else ''}")
    
    def count_current_request_tokens(self):
        if self.messages:
            user_msg = self.messages[-1] if self.messages[-1]["role"] == "user" else None
            if user_msg:
                return count_tokens(user_msg["content"], self.encoder)
        return 0
    
    def count_history_tokens(self):
        return count_messages_tokens(self.messages, self.encoder)
    
    def send_request(self, user_message):
        if self.use_compression and self.context_manager:
            self.context_manager.add_message("user", user_message)
            self.check_and_compress()
            messages_to_send = self.context_manager.build_messages(self.system_prompt)
        else:
            self.messages.append({"role": "user", "content": user_message})
            messages_to_send = self.messages
        
        payload = {
            "model": self.api_config.get("model", "deepseek-chat"),
            "messages": messages_to_send,
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
            
            if self.use_compression and self.context_manager:
                self.context_manager.add_message("assistant", assistant_reply)
            else:
                self.messages.append({"role": "assistant", "content": assistant_reply})
            
            usage = response.get('usage', {})
            api_prompt_tokens = usage.get('prompt_tokens', 0)
            api_completion_tokens = usage.get('completion_tokens', 0)
            api_total_tokens = usage.get('total_tokens', 0)
            
            self.total_tokens_used += api_total_tokens
            
            if self.use_compression and self.context_manager:
                local_history_tokens = count_messages_tokens(
                    self.context_manager.build_messages(self.system_prompt), 
                    self.encoder
                )
            else:
                local_history_tokens = self.count_history_tokens()
            
            local_response_tokens = count_tokens(assistant_reply, self.encoder)
            local_request_tokens = api_prompt_tokens - 4
            
            print("\n" + "=" * 40)
            print("📊 Token Usage:")
            print(f"  Current request:   {local_request_tokens} tokens")
            print(f"  Response:          {local_response_tokens} tokens")
            print(f"  History total:    {local_history_tokens} tokens")
            if self.use_compression and self.compression_count > 0:
                print(f"  Tokens saved:      {self.tokens_saved} (compressions: {self.compression_count})")
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
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens_used = 0
        self.tokens_saved = 0
        self.compression_count = 0
        if self.context_manager:
            self.context_manager = ContextManager(
                self.context_manager.keep_recent, 
                self.context_manager.compress_every
            )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepSync Terminal Chat")
    parser.add_argument("--compress", action="store_true", help="Enable context compression")
    parser.add_argument("--keep", type=int, default=10, help="Number of recent messages to keep (default: 10)")
    parser.add_argument("--compress-every", type=int, default=10, help="Compress after N messages (default: 10)")
    args = parser.parse_args()
    
    api_config = load_config()
    agent = Agent(api_config, use_compression=args.compress, keep_recent=args.keep, compress_every=args.compress_every)
    
    mode = "COMPRESSED" if args.compress else "FULL"
    print(f"DeepSync Terminal Chat [{mode} mode]")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    if args.compress:
        print(f"Compression: keep {args.keep} recent, compress every {args.compress_every} messages")
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
