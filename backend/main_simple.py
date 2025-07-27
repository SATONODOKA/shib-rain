from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import os
import requests
import json
import re
from datetime import datetime

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# データモデル
class ChatMessage(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    knowledge_files: List[Dict]
    chat_id: str

class IntentClarificationRequest(BaseModel):
    user_input: str
    chat_id: str

class IntentQuestion(BaseModel):
    question: str
    question_number: int
    max_questions: int
    is_complete: bool = False

class IntentClarificationResponse(BaseModel):
    question: IntentQuestion
    is_ambiguous: bool

class IntentResult(BaseModel):
    primary_intent: str
    keywords: List[str]
    knowledge_files: List[Dict]

class IntentQuestionRequest(BaseModel):
    user_input: str
    question_number: int
    answers: List[str] = []

class IntentCompleteRequest(BaseModel):
    user_input: str
    answers: List[str]
    chat_id: Optional[str] = None

# LLMサービス
class LLMService:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "qwen2.5:7b"
        self.is_available = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if self.model in model.get("name", ""):
                        print(f"Ollama is available with model: {self.model}")
                        return True
                print(f"Model {self.model} not found in available models")
                return False
        except Exception as e:
            print(f"Ollama service is not running. Please start Ollama first.")
            return False
        return False
    
    def generate_response(self, prompt: str) -> str:
        if not self.is_available:
            return "申し訳ございませんが、現在ローカルLLMが利用できません。"
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"LLM API error: {response.text}")
                return "LLMからの応答を取得できませんでした。"
                
        except Exception as e:
            print(f"LLM API error: {e}")
            return "LLMとの通信でエラーが発生しました。"

# ナレッジ管理
class KnowledgeManager:
    def __init__(self, knowledge_base_path: str = None):
        if knowledge_base_path is None:
            self.knowledge_base_path = Path("../KNOWLEDGE")
            if not self.knowledge_base_path.exists():
                self.knowledge_base_path = Path("KNOWLEDGE")
        else:
            self.knowledge_base_path = Path(knowledge_base_path)
        
        self.knowledge_files = []
        self.scan_knowledge_files()
    
    def scan_knowledge_files(self) -> List[Dict]:
        if not self.knowledge_base_path.exists():
            print(f"ナレッジベースパスが存在しません: {self.knowledge_base_path}")
            return []
        
        files = []
        for category_dir in self.knowledge_base_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for file_path in category_dir.glob("*.md"):
                    file_info = self._parse_knowledge_file(file_path, category)
                    if file_info:
                        files.append(file_info)
        
        self.knowledge_files = files
        print(f"Scanned {len(files)} knowledge files")
        return files
    
    def _parse_knowledge_file(self, file_path: Path, category: str) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title = file_path.stem
            rel_path = f"{category}/{file_path.name}"
            
            return {
                "path": rel_path,
                "title": title,
                "category": category,
                "keywords": [],
                "description": content[:200] + "..." if len(content) > 200 else content,
                "file_size": len(content),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "score": 0,
                "content": content
            }
        except Exception as e:
            print(f"ファイル解析エラー {file_path}: {e}")
            return None
    
    def get_knowledge_files(self) -> List[Dict]:
        return self.knowledge_files

# 意図特定サービス
class IntentClarificationService:
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
        self.llm_service = LLMService()
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> IntentQuestion:
        if answers is None:
            answers = []
        
        # LLMに質問生成を任せる
        prompt = f"""
ユーザーの質問「{user_input}」について、知りたい具体的な情報を確認する質問を1つ生成してください。
これまでの回答: {', '.join(answers) if answers else 'なし'}
質問番号: {question_number}/3

質問のみを返してください。もう十分な情報がある場合は「COMPLETE」と返してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt).strip()
            
            if "COMPLETE" in response.upper():
                return IntentQuestion(
                    question="理解しました。回答を生成します。",
                    question_number=question_number,
                    max_questions=3,
                    is_complete=True
                )
            
            return IntentQuestion(
                question=response,
                question_number=question_number,
                max_questions=3
            )
        except Exception as e:
            # フォールバック
            question = f"「{user_input}」について、どのような具体的な情報をお探しですか？"
            return IntentQuestion(
                question=question,
                question_number=question_number,
                max_questions=3
            )
    
    def identify_intent(self, user_input: str, answers: List[str]) -> IntentResult:
        combined_input = f"{user_input} {' '.join(answers)}"
        
        # LLMに意図特定とナレッジ検索を任せる
        prompt = f"""
ユーザーの質問から、検索に使用するキーワードを抽出してください。

質問: {combined_input}

以下のJSON形式で回答してください：
{{
    "primary_intent": "主要な意図",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"]
}}

JSONのみを返してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                primary_intent = data.get("primary_intent", "一般情報")
                keywords = data.get("keywords", [])
            else:
                primary_intent = "一般情報"
                keywords = []
        except:
            primary_intent = "一般情報"
            keywords = []
        
        # ナレッジファイルを検索（LLMベース）
        knowledge_files = self._search_knowledge_with_llm(combined_input)
        
        return IntentResult(
            primary_intent=primary_intent,
            keywords=keywords,
            knowledge_files=knowledge_files
        )
    
    def _search_knowledge_with_llm(self, query: str) -> List[Dict]:
        files = self.knowledge_manager.get_knowledge_files()
        
        prompt = f"""
ユーザーの質問「{query}」に関連するファイルを選んでください。

利用可能なファイル:
{chr(10).join([f"- {file['title']} ({file['path']})" for file in files])}

関連度の高い順に、最大5つのファイルを選んでください。
以下のJSON形式で回答してください：

{{
    "selected_files": [
        {{
            "title": "ファイルタイトル"
        }}
    ]
}}

JSONのみを返してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                selected_files = data.get("selected_files", [])
                
                results = []
                for selected in selected_files:
                    for file_info in files:
                        if file_info['title'] == selected['title']:
                            results.append({
                                'path': file_info['path'],
                                'title': file_info['title'],
                                'category': file_info.get('category', ''),
                                'keywords': file_info.get('keywords', []),
                                'description': file_info.get('description', ''),
                                'file_size': file_info.get('file_size', 0),
                                'last_modified': file_info.get('last_modified', ''),
                                'score': 100
                            })
                            break
                
                return results
            else:
                return []
        except:
            return []

# グローバル変数としてサービスを定義
knowledge_manager = None
intent_clarification_service = None

# APIエンドポイント
@app.get("/")
async def root():
    return {"message": "Knowledge Chat API - Simple Version"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "knowledge_files_count": len(knowledge_manager.get_knowledge_files())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        knowledge_files = intent_clarification_service._search_knowledge_with_llm(message.message)
        response = await _generate_response(message.message, knowledge_files)
        
        return ChatResponse(
            response=response,
            knowledge_files=knowledge_files,
            chat_id=message.chat_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_response(user_message: str, knowledge_files: List[Dict]) -> str:
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    # LLMに回答生成を任せる
    file_contents = []
    for file in knowledge_files[:3]:
        try:
            file_path = Path("../KNOWLEDGE") / file['path']
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                file_contents.append(f"【{file['title']}】\n{content[:1000]}...")
        except:
            continue
    
    if not file_contents:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    prompt = f"""
ユーザーの質問「{user_message}」について、以下のナレッジファイルの情報を基に回答してください。

ナレッジファイルの内容:
{chr(10).join(file_contents)}

自然で分かりやすい回答を生成してください。
"""
    
    try:
        response = intent_clarification_service.llm_service.generate_response(prompt)
        return response
    except:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりましたが、回答の生成に失敗しました。"

@app.post("/intent/start", response_model=IntentClarificationResponse)
async def start_intent_clarification(request: IntentClarificationRequest):
    try:
        question = intent_clarification_service.generate_question(request.user_input, 1)
        return IntentClarificationResponse(
            question=question,
            is_ambiguous=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/question")
async def generate_next_question(request: IntentQuestionRequest):
    try:
        question = intent_clarification_service.generate_question(
            request.user_input, 
            request.question_number, 
            request.answers
        )
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete", response_model=IntentResult)
async def complete_intent_clarification(request: IntentCompleteRequest):
    try:
        intent_result = intent_clarification_service.identify_intent(request.user_input, request.answers)
        return intent_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete-with-response", response_model=ChatResponse)
async def complete_intent_with_response(request: IntentCompleteRequest):
    try:
        intent_result = intent_clarification_service.identify_intent(request.user_input, request.answers)
        combined_query = f"{request.user_input} {' '.join(request.answers)}"
        response = await _generate_response(combined_query, intent_result.knowledge_files)
        
        return ChatResponse(
            response=response,
            knowledge_files=intent_result.knowledge_files,
            chat_id=request.chat_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/search-by-intent")
async def search_knowledge_by_intent(intent_result: IntentResult):
    try:
        return {
            "knowledge_files": intent_result.knowledge_files,
            "intent": intent_result.primary_intent,
            "keywords": intent_result.keywords,
            "explanation": f"あなたの意図「{intent_result.primary_intent}」に基づいて、関連するナレッジを表示しています。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# グローバル変数としてサービスを定義
knowledge_manager = KnowledgeManager()
intent_clarification_service = IntentClarificationService(knowledge_manager)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 