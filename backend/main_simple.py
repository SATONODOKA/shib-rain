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

# データモデル（要件定義書に基づく）
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
    is_complete: bool = False

class IntentClarificationResponse(BaseModel):
    question: IntentQuestion
    is_ambiguous: bool

class UserNeeds(BaseModel):
    description: str
    keywords: List[str]

class ObsidianSearchRequest(BaseModel):
    user_needs: UserNeeds

class ObsidianSearchResult(BaseModel):
    files: List[Dict]
    explanation: str

# LLMサービス（要件定義書1.4節に基づく）
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
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
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
                "description": content[:200] + "..." if len(content) > 200 else content,
                "file_size": len(content),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "content": content
            }
        except Exception as e:
            print(f"ファイル解析エラー {file_path}: {e}")
            return None
    
    def get_knowledge_files(self) -> List[Dict]:
        return self.knowledge_files

# 意図特定サービス（要件定義書1.4節に基づく）
class IntentClarificationService:
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
        self.llm_service = LLMService()
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> IntentQuestion:
        if answers is None:
            answers = []
        
        # 要件定義書3.1.1に基づくシンプルなプロンプト
        prompt = f"""
あなたはユーザーのニーズを質問を通じて具体化したうえで、Obsidian上の関連性の高いファイルやナレッジを5つほど選び、表示させてください。

ユーザー入力: {user_input}
現在の質問回数: {question_number}
これまでの回答: {', '.join(answers) if answers else 'なし'}

質問が必要な場合は、ユーザーのニーズを具体化するための質問を1つ生成してください。
十分な情報が得られた場合は、関連性の高いナレッジファイルを5つ程度選択して表示してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt).strip()
            
            # LLMが十分な情報を得たと判断した場合
            if any(keyword in response.lower() for keyword in ['理解しました', '検索します', '関連ファイル', 'ナレッジファイル']):
                return IntentQuestion(
                    question="理解しました。関連するナレッジファイルを検索します。",
                    question_number=question_number,
                    is_complete=True
                )
            
            return IntentQuestion(
                question=response,
                question_number=question_number
            )
        except Exception as e:
            # フォールバック（制約最小化）
            question = f"「{user_input}」について、より詳しい情報をお聞かせください。"
            return IntentQuestion(
                question=question,
                question_number=question_number
            )
    
    def search_obsidian_with_llm(self, user_needs: str) -> List[Dict]:
        files = self.knowledge_manager.get_knowledge_files()
        
        if not files:
            print("ナレッジファイルが見つかりません")
            return []
        
        # 要件定義書3.1.2に基づく検索実行プロンプト
        prompt = f"""
ユーザーのニーズに基づいて、Obsidianで関連性の高いファイルを5つ程度検索してください。

ユーザーニーズ: {user_needs}

利用可能なファイル:
{chr(10).join([f"- {file['title']} ({file['path']})" for file in files])}

以下の形式で検索結果を返してください:
{{
  "files": [
    {{
      "path": "ファイルパス",
      "title": "ファイルタイトル",
      "relevance": "関連性の説明"
    }}
  ],
  "explanation": "検索の説明"
}}
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            print(f"LLM応答: {response}")
            
            # LLMがエラーメッセージを返した場合はフォールバック
            if any(error_keyword in response for error_keyword in ['エラー', '取得できません', '通信でエラー']):
                print("LLMエラー検出、フォールバック検索を実行")
                fallback_results = self._fallback_search(user_needs, files)
                if fallback_results:
                    print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
                return fallback_results
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                selected_files = data.get("files", [])
                
                results = []
                for selected in selected_files:
                    for file_info in files:
                        if file_info['title'] == selected['title']:
                            results.append({
                                'path': file_info['path'],
                                'title': file_info['title'],
                                'category': file_info.get('category', ''),
                                'description': file_info.get('description', ''),
                                'relevance': selected.get('relevance', '関連性の高いファイル'),
                                'file_size': file_info.get('file_size', 0),
                                'last_modified': file_info.get('last_modified', '')
                            })
                            break
                
                if results:
                    print(f"LLM検索で{len(results)}個のファイルを発見")
                    return results
                else:
                    # LLMがファイルを選択しなかった場合のフォールバック
                    print("LLMがファイルを選択しませんでした、フォールバック検索を実行")
                    fallback_results = self._fallback_search(user_needs, files)
                    if fallback_results:
                        print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
                    return fallback_results
            else:
                # JSON形式で応答しなかった場合のフォールバック
                print("LLMがJSON形式で応答しませんでした、フォールバック検索を実行")
                fallback_results = self._fallback_search(user_needs, files)
                if fallback_results:
                    print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
                return fallback_results
        except Exception as e:
            print(f"LLM検索エラー: {e}")
            # エラー時のフォールバック
            fallback_results = self._fallback_search(user_needs, files)
            if fallback_results:
                print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
            return fallback_results
    
    def _fallback_search(self, user_needs: str, files: List[Dict]) -> List[Dict]:
        """フォールバック検索: キーワードベースの検索"""
        print(f"フォールバック検索実行: {user_needs}")
        
        # ユーザーニーズからキーワードを抽出
        keywords = self._extract_keywords(user_needs)
        
        results = []
        for file_info in files:
            score = 0
            
            # タイトルでのマッチング
            title_lower = file_info['title'].lower()
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    score += 10
            
            # パスでのマッチング
            path_lower = file_info['path'].lower()
            for keyword in keywords:
                if keyword.lower() in path_lower:
                    score += 5
            
            # 説明でのマッチング
            description_lower = file_info.get('description', '').lower()
            for keyword in keywords:
                if keyword.lower() in description_lower:
                    score += 3
            
            if score > 0:
                results.append({
                    'path': file_info['path'],
                    'title': file_info['title'],
                    'category': file_info.get('category', ''),
                    'description': file_info.get('description', ''),
                    'relevance': f'キーワードマッチング (スコア: {score})',
                    'file_size': file_info.get('file_size', 0),
                    'last_modified': file_info.get('last_modified', '')
                })
        
        # スコアでソートして上位5件を返す
        results.sort(key=lambda x: int(x['relevance'].split('スコア: ')[1].split(')')[0]), reverse=True)
        return results[:5]
    
    def _extract_keywords(self, user_needs: str) -> List[str]:
        """ユーザーニーズからキーワードを抽出"""
        keywords = []
        
        # 基本的なキーワード抽出
        if '中堅' in user_needs or '中堅層' in user_needs:
            keywords.extend(['中堅', '中堅層', '若手中堅離職防止'])
        
        if '離職' in user_needs:
            keywords.extend(['離職', '離職防止', '定着', '若手中堅離職防止'])
        
        if '建設' in user_needs:
            keywords.extend(['建設業界', '建設'])
        
        if '製造' in user_needs:
            keywords.extend(['製造業界', '製造'])
        
        if 'IT' in user_needs or 'it' in user_needs.lower():
            keywords.extend(['IT業界', 'IT'])
        
        if '組織風土' in user_needs:
            keywords.extend(['組織風土改革', '組織風土'])
        
        if '女性' in user_needs:
            keywords.extend(['女性活躍', '女性'])
        
        if '働き方' in user_needs:
            keywords.extend(['働き方改革', '働き方'])
        
        # デフォルトキーワード
        if not keywords:
            keywords = ['中堅', '離職', '若手中堅離職防止']
        
        return keywords

# グローバル変数としてサービスを定義
print("KnowledgeManager初期化開始...")
knowledge_manager = KnowledgeManager()
print(f"KnowledgeManager初期化完了: {len(knowledge_manager.get_knowledge_files())}個のファイルを読み込み")

print("IntentClarificationService初期化開始...")
intent_clarification_service = IntentClarificationService(knowledge_manager)
print("IntentClarificationService初期化完了")

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
        # LLMに直接検索を委任
        knowledge_files = intent_clarification_service.search_obsidian_with_llm(message.message)
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
    
    # LLMに回答生成を委任（制約最小化）
    file_contents = []
    for file in knowledge_files[:3]:
        try:
            file_path = Path("../KNOWLEDGE") / file['path']
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                file_contents.append(f"【{file['title']}】\n{content[:1000]}...")
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            continue
    
    if not file_contents:
        # ファイルが見つかったが内容を読み込めなかった場合
        return f"「{user_message}」について、関連するナレッジファイルが見つかりましたが、内容の読み込みに失敗しました。"
    
    # シンプルなプロンプト（制約最小化）
    prompt = f"""
ユーザーの質問「{user_message}」について、以下のナレッジファイルの情報を基に回答してください。

ナレッジファイルの内容:
{chr(10).join(file_contents)}

自然で分かりやすい回答を生成してください。
"""
    
    try:
        response = intent_clarification_service.llm_service.generate_response(prompt)
        if response and response.strip() and not any(error_keyword in response for error_keyword in ['エラー', '取得できません', '通信でエラー']):
            return response
        else:
            # LLMが応答しなかった場合のフォールバック
            print("LLM応答生成に失敗、フォールバック応答を生成")
            return _generate_fallback_response(user_message, knowledge_files)
    except Exception as e:
        print(f"LLM応答生成エラー: {e}")
        # エラー時のフォールバック
        return _generate_fallback_response(user_message, knowledge_files)

def _generate_fallback_response(user_message: str, knowledge_files: List[Dict]) -> str:
    """フォールバック応答生成"""
    print(f"フォールバック応答生成: {user_message}")
    
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    response = f"「{user_message}」について、以下のナレッジファイルが関連していると思われます：\n\n"
    
    for i, file in enumerate(knowledge_files[:3], 1):
        response += f"{i}. **{file['title']}**\n"
        response += f"   - カテゴリ: {file['category']}\n"
        response += f"   - 関連性: {file.get('relevance', '関連性の高いファイル')}\n"
        response += f"   - 概要: {file.get('description', '')[:100]}...\n\n"
    
    response += "これらのファイルをObsidianで開いて詳細を確認してください。"
    return response

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
async def generate_next_question(request: IntentClarificationRequest):
    try:
        question = intent_clarification_service.generate_question(
            request.user_input, 
            1  # 文脈に応じて変動（要件定義書9.2節）
        )
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete", response_model=ObsidianSearchResult)
async def complete_intent_clarification(request: IntentClarificationRequest):
    try:
        # LLMに直接検索を委任
        knowledge_files = intent_clarification_service.search_obsidian_with_llm(request.user_input)
        
        return ObsidianSearchResult(
            files=knowledge_files,
            explanation="ユーザーのニーズに基づいて関連性の高いナレッジファイルを検索しました。"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete-with-response", response_model=ChatResponse)
async def complete_intent_with_response(request: IntentClarificationRequest):
    try:
        # LLMに直接検索を委任
        knowledge_files = intent_clarification_service.search_obsidian_with_llm(request.user_input)
        response = await _generate_response(request.user_input, knowledge_files)
        
        return ChatResponse(
            response=response,
            knowledge_files=knowledge_files,
            chat_id=request.chat_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/search-obsidian", response_model=ObsidianSearchResult)
async def search_obsidian(request: ObsidianSearchRequest):
    try:
        # LLMに直接検索を委任
        knowledge_files = intent_clarification_service.search_obsidian_with_llm(request.user_needs.description)
        
        return ObsidianSearchResult(
            files=knowledge_files,
            explanation="Obsidianで関連性の高いナレッジファイルを検索しました。"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("FastAPIサーバー起動中...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 