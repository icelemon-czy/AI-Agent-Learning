import requests
import json
import sys

class OllamaClient:
    def __init__(self, host="http://localhost:11434"):
        self.base_url = host.rstrip('/')
        self.models_url = f"{self.base_url}/api/tags"
        self.generate_url = f"{self.base_url}/api/generate"
        
    def get_available_models(self):
        """获取所有可用模型"""
        try:
            response = requests.get(self.models_url)
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                return [model['name'] for model in models_data]
            else:
                return []
        except requests.exceptions.RequestException as e:
            print(f"连接模型服务器失败: {e}")
            sys.exit(1)
    
    def generate_response(self, model_name, prompt, stream=False):
        """生成模型响应"""
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream
        }
        
        try:
            response = requests.post(self.generate_url, json=data, stream=stream)
            if response.status_code == 200:
                if stream:
                    # 处理流式响应
                    return self._handle_stream_response(response)
                else:
                    # 处理非流式响应
                    result = response.json()
                    return result
            else:
                return {"error": f"API 请求失败: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"请求错误: {str(e)}"}
    
    def chat(self, model_name, context_window=3):
        """与模型进行交互式对话"""
        print(f"正在使用模型: {model_name}")
        print("请输入您的问题 (输入 'exit' 退出):")
        
        while True:
            user_input = input("\n您: ")
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("再见！")
                break
                
            response = self.generate_response(model_name, user_input)
            
            if 'error' in response:
                print(f"错误: {response['error']}")
            else:
                if response.get('stream', False):
                    # 流式已在 generate_response 中处理
                    pass
                else:
                    self._handle_single_response(response)
                    
                # 保存对话历史
                history_file = f"{model_name}_chat_history.txt"
                with open(history_file, 'a', encoding='utf-8') as f:
                    f.write(f"用户: {user_input}\n")
                    f.write(f"助手: {response.get('response', '')}\n\n")
    
    def _handle_stream_response(self, response):
        """处理流式响应"""
        response_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'response' in chunk:
                    response_text += chunk['response']
                    print(chunk['response'], end='', flush=True)
                if chunk.get('done', False):
                    print("\n")
                    break
        return {"response": response_text}
    
    def _handle_single_response(self, response):
        """处理非流式响应"""
        response_text = response.get('response', '')
        print(f"{response_text}\n")
        
        # 处理模型返回的候选内容（如果有）
        candidates = response.get('candidates', [])
        for candidate in candidates:
            if 'content' in candidate:
                print(f"候选内容: {candidate['content']}")
        
        return response_text

if __name__ == "__main__":
    # 创建 Ollama 客户端实例
    ollama_client = OllamaClient()
    
    # 获取可用模型
    available_models = ollama_client.get_available_models()
    if not available_models:
        print("没有可用的模型，请确保 Ollama 服务正在运行并已加载模型")
        sys.exit(1)
    
    print("可用模型:")
    for model in available_models:
        print(f"- {model}")
    
    # 选择模型进行对话
    selected_model = input("\n请选择一个模型进行对话: ")
    if selected_model not in available_models:
        print("未找到指定的模型，请检查模型名称是否正确")
        sys.exit(1)
    
    # 开始与模型对话
    ollama_client.chat(selected_model)