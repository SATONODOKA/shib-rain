from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    needs_clarification: bool = False

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
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "num_thread": 4,
                        "temperature": 0.7
                    }
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
    
    def needs_clarification(self, user_input: str) -> bool:
        """質問の掘り下げが必要かどうかを判断"""
        # 十分具体的な質問の場合は掘り下げ不要
        specific_keywords = ['エンゲージメント', '離職', '組織風土', '人材育成', 'コミュニケーション', 'ハラスメント', '女性活躍', 'デジタル化', '働き方改革']
        
        if any(keyword in user_input for keyword in specific_keywords):
            return False
        
        prompt = f"""
あなたは日本語で回答するアシスタントです。

ユーザーの質問「{user_input}」について、掘り下げが必要かどうかを判断してください。

掘り下げが必要な場合のみ「はい」と回答してください。
それ以外は「いいえ」と回答してください。

「はい」または「いいえ」のみで回答してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt).strip().lower()
            # より厳密な判断：明確に「はい」と回答した場合のみ掘り下げ
            return response in ['はい', 'yes', '必要', 'true', '1']
        except Exception as e:
            print(f"掘り下げ判断エラー: {e}")
            # エラーの場合は安全のため掘り下げを提案
            return True
    
    def generate_clarification_question(self, user_input: str) -> str:
        """質問の掘り下げを行う自然な質問を生成"""
        prompt = f"""
重要: あなたは日本語で回答するアシスタントです。英語や中国語は一切使用せず、必ず日本語のみで回答してください。

ユーザーの質問「{user_input}」について、より具体的な情報を得るための自然な質問を1つ生成してください。

質問の特徴:
- ユーザーの興味関心を動的に特定する
- 人間らしい自然な会話を心がける
- 固定パターンに縛られない
- 最終的にはObsidianのナレッジファイルに導く

必ず日本語のみで回答してください。英語や中国語の単語や表現は一切使用しないでください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt).strip()
            if response and len(response) > 10:
                return response
            else:
                return f"「{user_input}」について、もう少し詳しく教えていただけますか？"
        except Exception as e:
            print(f"質問生成エラー: {e}")
            return f"「{user_input}」について、もう少し詳しく教えていただけますか？"
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> IntentQuestion:
        if answers is None:
            answers = []
        
        # 動的・人間らしい対話のためのプロンプト
        prompt = f"""
あなたは、ユーザーの興味関心を自然な対話を通じて理解し、Obsidianにあるナレッジやメモに導くアシスタントです。

ユーザー入力: {user_input}
現在の質問回数: {question_number}
これまでの回答: {', '.join(answers) if answers else 'なし'}

対話の特徴:
- ユーザーの興味関心を動的に特定する
- 人間らしい自然な会話を心がける
- 固定パターンや強制的な質問回数に縛られない
- 最終的にはObsidianのナレッジファイルに導く

質問が必要な場合は、ユーザーの興味関心をより深く理解するための自然な質問を1つ生成してください。
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
        
        # メモリ最適化: ファイルリストを最初の50個に制限
        limited_files = files[:50]
        
        # 効率的な検索のためのプロンプト
        prompt = f"""
ユーザーの質問「{user_needs}」に関連するファイルを5つ選んでください。

利用可能なファイル:
{chr(10).join([f"- {file['title']}" for file in limited_files])}

以下の形式で返してください:
{{
  "files": [
    {{"title": "ファイルタイトル", "relevance": "関連性"}}
  ]
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
                added_titles = set()  # 重複防止のためのセット
                for selected in selected_files:
                    # 既に追加されたファイルはスキップ
                    if selected.get('title') in added_titles:
                        continue
                        
                    # メインLLM検索: 全ファイルから詳細情報を取得
                    for file_info in files:
                        if file_info['title'] == selected.get('title'):
                            results.append({
                                'path': file_info['path'],
                                'title': file_info['title'],
                                'category': file_info.get('category', ''),
                                'description': file_info.get('description', ''),
                                'relevance': selected.get('relevance', '関連性の高いファイル'),
                                'file_size': file_info.get('file_size', 0),
                                'last_modified': file_info.get('last_modified', '')
                            })
                            added_titles.add(file_info['title'])  # 追加済みとしてマーク
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
        """フォールバック検索: LLMによる柔軟な検索"""
        print(f"フォールバック検索実行: {user_needs}")
        
        # メモリ最適化: ファイルリストを最初の50個に制限
        limited_files = files[:50]
        
        # 簡潔なフォールバック検索のプロンプト
        prompt = f"""
「{user_needs}」に関連するファイルを5つ選んでください。

利用可能なファイル:
{chr(10).join([f"- {file['title']}" for file in limited_files])}

以下の形式で返してください:
{{
  "files": [
    {{"title": "ファイルタイトル", "relevance": "関連性"}}
  ]
}}
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            print(f"フォールバックLLM応答: {response}")
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                selected_files = data.get("files", [])
                
                results = []
                added_titles = set()  # 重複防止のためのセット
                for selected in selected_files:
                    # 既に追加されたファイルはスキップ
                    if selected.get('title') in added_titles:
                        continue
                        
                    # フォールバック検索: 全ファイルから詳細情報を取得
                    for file_info in files:
                        if file_info['title'] == selected.get('title'):
                            results.append({
                                'path': file_info['path'],
                                'title': file_info['title'],
                                'category': file_info.get('category', ''),
                                'description': file_info.get('description', ''),
                                'relevance': selected.get('relevance', 'フォールバック検索による関連ファイル'),
                                'file_size': file_info.get('file_size', 0),
                                'last_modified': file_info.get('last_modified', '')
                            })
                            added_titles.add(file_info['title'])  # 追加済みとしてマーク
                            break
                
                if results:
                    print(f"フォールバック検索で{len(results)}個のファイルを発見")
                    return results
            
            # LLMが応答しなかった場合の最終フォールバック
            print("フォールバックLLMも応答せず、最終フォールバックを実行")
            return self._final_fallback_search(user_needs, files)
            
        except Exception as e:
            print(f"フォールバック検索エラー: {e}")
            return self._final_fallback_search(user_needs, files)
    
    def _final_fallback_search(self, user_needs: str, files: List[Dict]) -> List[Dict]:
        """最終フォールバック: 最もシンプルな検索"""
        print(f"最終フォールバック検索実行: {user_needs}")
        
        # 最もシンプルな検索 - ファイル名にユーザー入力の一部が含まれるものを返す
        results = []
        added_titles = set()  # 重複防止のためのセット
        user_words = user_needs.split()
        
        for file_info in files:
            # 既に追加されたファイルはスキップ
            if file_info['title'] in added_titles:
                continue
                
            title_lower = file_info['title'].lower()
            path_lower = file_info['path'].lower()
            
            for word in user_words:
                if len(word) > 1 and (word.lower() in title_lower or word.lower() in path_lower):
                    results.append({
                        'path': file_info['path'],
                        'title': file_info['title'],
                        'category': file_info.get('category', ''),
                        'description': file_info.get('description', ''),
                        'relevance': f'最終フォールバック検索 (マッチ: {word})',
                        'file_size': file_info.get('file_size', 0),
                        'last_modified': file_info.get('last_modified', '')
                    })
                    added_titles.add(file_info['title'])  # 追加済みとしてマーク
                    break
        
        return results[:5]

# グローバル変数としてサービスを定義
print("KnowledgeManager初期化開始...")
knowledge_manager = KnowledgeManager()
print(f"KnowledgeManager初期化完了: {len(knowledge_manager.get_knowledge_files())}個のファイルを読み込み")

print("IntentClarificationService初期化開始...")
intent_clarification_service = IntentClarificationService(knowledge_manager)
print("IntentClarificationService初期化完了")

# APIエンドポイント
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # まず質問の掘り下げが必要かどうかを判断
        clarification_needed = intent_clarification_service.needs_clarification(message.message)
        
        if clarification_needed:
            # 質問の掘り下げが必要な場合
            question = intent_clarification_service.generate_clarification_question(message.message)
            return ChatResponse(
                response=question,
                knowledge_files=[],
                chat_id=message.chat_id or "default",
                needs_clarification=True
            )
        else:
            # 十分な情報がある場合は直接検索
            knowledge_files = intent_clarification_service.search_obsidian_with_llm(message.message)
            response = await _generate_response(message.message, knowledge_files)
            
            return ChatResponse(
                response=response,
                knowledge_files=knowledge_files,
                chat_id=message.chat_id or "default",
                needs_clarification=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 静的ファイル提供設定
app.mount("/static", StaticFiles(directory="../FRONTEND"), name="static")

@app.get("/")
async def read_index():
    """フロントエンドのindex.htmlを提供"""
    return FileResponse('../FRONTEND/index.html')

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "knowledge_files_count": len(knowledge_manager.get_knowledge_files())
    }

async def _generate_response(user_message: str, knowledge_files: List[Dict]) -> str:
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    # 既に読み込まれているファイル内容を使用（高速化）
    file_contents = []
    for file in knowledge_files[:3]:
        try:
            # KnowledgeManagerで既に読み込まれているcontentを使用
            content = file.get('content', '')
            if content:
                file_contents.append(f"【{file['title']}】\n{content[:500]}...")
            else:
                # contentがない場合はdescriptionを使用
                description = file.get('description', '')
                if description:
                    file_contents.append(f"【{file['title']}】\n{description[:200]}...")
        except Exception as e:
            print(f"ファイル内容取得エラー: {e}")
            continue
    
    if not file_contents:
        # ファイルが見つかったが内容を取得できなかった場合
        return f"「{user_message}」について、関連するナレッジファイルが見つかりましたが、内容の取得に失敗しました。"
    
    # 簡潔で効率的な回答生成のためのプロンプト
    prompt = f"""
重要: あなたは日本語で回答するアシスタントです。英語や中国語は一切使用せず、必ず日本語のみで回答してください。

ユーザーの質問「{user_message}」について、以下の情報を基に簡潔に回答してください。

{chr(10).join(file_contents[:2])}

必ず日本語のみで回答してください。英語や中国語の単語や表現は一切使用しないでください。
"""
    
    try:
        response = intent_clarification_service.llm_service.generate_response(prompt)
        if response and response.strip():
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
            explanation="ユーザーの興味関心に基づいて関連性の高いナレッジファイルを検索しました。"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete-with-response", response_model=ChatResponse)
async def complete_intent_with_response(request: IntentClarificationRequest):
    try:
        # LLMに直接検索を委任
        knowledge_files = intent_clarification_service.search_obsidian_with_llm(request.user_input)
        
        # 動的・人間らしい回答生成
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
            explanation="ユーザーの興味関心に基づいてObsidianで関連性の高いナレッジファイルを検索しました。"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("FastAPIサーバー起動中...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 