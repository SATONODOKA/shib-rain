#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ナレッジチャットシステム バックエンドAPI
ローカルLLMを使用してユーザーの曖昧な投げかけを理解し、関連性の高いナレッジを特定する
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

app = FastAPI(title="ナレッジチャットAPI", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの提供
app.mount("/static", StaticFiles(directory="../FRONTEND"), name="static")

# データモデル
class ChatMessage(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    knowledge_files: List[Dict]
    chat_id: str

class KnowledgeFile(BaseModel):
    path: str
    title: str
    category: str
    keywords: List[str]
    description: str
    score: float

# 意図特定機能のデータモデル
class IntentClarificationRequest(BaseModel):
    user_input: str
    chat_id: str

class IntentQuestion(BaseModel):
    question: str
    options: List[str]
    question_number: int
    max_questions: int

class IntentClarificationResponse(BaseModel):
    question: IntentQuestion
    is_ambiguous: bool

class IntentResult(BaseModel):
    primary_intent: str
    secondary_intent: Dict[str, str]
    confidence: float
    keywords: List[str]
    knowledge_files: List[Dict]

class IntentQuestionRequest(BaseModel):
    user_input: str
    question_number: int
    answers: List[str] = []

class IntentCompleteRequest(BaseModel):
    user_input: str
    answers: List[str]

# ナレッジ管理
class KnowledgeManager:
    def __init__(self, knowledge_base_path: str = "../KNOWLEDGE"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_files = []
        self.last_scan_time = None
        self.scan_knowledge_files()
    
    def scan_knowledge_files(self) -> List[Dict]:
        """ナレッジファイルをスキャン"""
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
        """ナレッジファイルを解析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ファイル名からタイトルを生成
            title = file_path.stem
            
            # キーワードを抽出（簡単な方法）
            keywords = self._extract_keywords(content)
            
            return {
                "path": str(file_path.relative_to(self.knowledge_base_path.parent)),
                "title": title,
                "category": category,
                "keywords": keywords,
                "description": content[:200] + "..." if len(content) > 200 else content,
                "file_size": len(content),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "score": 0
            }
        except Exception as e:
            print(f"ファイル解析エラー {file_path}: {e}")
            return None
    
    def _extract_keywords(self, content: str) -> List[str]:
        """簡単なキーワード抽出"""
        # 基本的なキーワード
        keywords = []
        
        # 建設業界関連
        if "建設" in content:
            keywords.append("建設業界")
        if "現場" in content:
            keywords.append("現場")
        if "職人" in content:
            keywords.append("職人")
        
        # 能力開発関連
        if "女性" in content:
            keywords.append("女性活躍")
        if "組織" in content:
            keywords.append("組織風土")
        if "パワハラ" in content or "ハラスメント" in content:
            keywords.append("パワハラ防止")
        if "研修" in content:
            keywords.append("研修")
        
        return keywords
    
    def get_knowledge_files(self) -> List[Dict]:
        """ナレッジファイル一覧を取得"""
        return self.knowledge_files

# LLMサービス
class LLMService:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "qwen2.5:7b"
        self.is_available = False  # LLMを無効化
    
    def _check_ollama_availability(self) -> bool:
        """Ollamaの利用可能性をチェック"""
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
        """LLMから応答を生成"""
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

# 意図特定サービス
class IntentClarificationService:
    def __init__(self, llm_service: LLMService, knowledge_manager: KnowledgeManager):
        self.llm_service = llm_service
        self.knowledge_manager = knowledge_manager
    
    def is_ambiguous(self, user_input: str) -> bool:
        """入力が曖昧かどうかを判定"""
        # 簡単な判定ロジック
        ambiguous_words = ["知りたい", "教えて", "について", "どう", "何か"]
        return any(word in user_input for word in ambiguous_words)
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> IntentQuestion:
        """質問を生成（シンプルなロジック）"""
        if answers is None:
            answers = []
        
        # ユーザーの回答を考慮した質問生成
        context = f"{user_input}"
        if answers:
            context += f" {' '.join(answers)}"
        
        if question_number == 1:
            question = f"「{user_input}」について、どのような情報をお探しですか？"
        elif question_number == 2:
            # 回答履歴を考慮
            if "パワハラ" in context or "撲滅" in context:
                question = "パワハラ防止について、どのような具体的な情報をお探しですか？"
            elif "建設" in context:
                question = "建設業界のどのような側面について知りたいですか？"
            else:
                question = "どのような立場の方向けの情報をお探しですか？"
        else:
            question = "どのような段階の情報をお探しですか？"
        
        return IntentQuestion(
            question=question,
            question_number=question_number,
            max_questions=3
        )
    
    def identify_intent(self, user_input: str, answers: List[str]) -> IntentResult:
        """意図を特定"""
        # 結合された入力からキーワードを抽出
        combined_input = f"{user_input} {' '.join(answers)}"
        
        # キーワード抽出
        keywords = self._extract_keywords(combined_input)
        
        # 意図を判定
        primary_intent = self._determine_intent(combined_input)
        
        # ナレッジファイルを検索
        knowledge_files = self._search_knowledge(combined_input)
        
        return IntentResult(
            primary_intent=primary_intent,
            keywords=keywords,
            knowledge_files=knowledge_files
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """キーワードを抽出"""
        keywords = []
        
        # 基本的なキーワード抽出
        if "パワハラ" in text or "ハラスメント" in text:
            keywords.append("パワハラ防止")
        if "建設" in text:
            keywords.append("建設業界")
        if "研修" in text:
            keywords.append("研修")
        if "女性" in text:
            keywords.append("女性活躍")
        if "組織" in text:
            keywords.append("組織風土")
        if "現場" in text:
            keywords.append("現場")
        
        return keywords
    
    def _determine_intent(self, text: str) -> str:
        """意図を判定"""
        if "パワハラ" in text or "ハラスメント" in text:
            return "パワハラ防止"
        elif "建設" in text:
            return "建設業界"
        elif "研修" in text:
            return "研修実施"
        else:
            return "一般情報"
    
    def _search_knowledge(self, query: str) -> List[Dict]:
        """ナレッジを検索"""
        files = self.knowledge_manager.get_knowledge_files()
        results = []
        
        query_lower = query.lower()
        
        for file in files:
            score = 0
            
            # タイトルでマッチング
            if any(keyword.lower() in file["title"].lower() for keyword in query_lower.split()):
                score += 5
            
            # キーワードでマッチング
            for keyword in file["keywords"]:
                if keyword.lower() in query_lower:
                    score += 3
            
            # 説明文でマッチング
            if any(keyword.lower() in file["description"].lower() for keyword in query_lower.split()):
                score += 1
            
            if score > 0:
                file["score"] = score
                results.append(file)
        
        # スコアでソート
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]  # 上位5件を返す

# サービスインスタンス
knowledge_manager = KnowledgeManager()
llm_service = LLMService()
intent_clarification_service = IntentClarificationService(llm_service, knowledge_manager)

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Knowledge Chat API"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "llm_available": llm_service.is_available,
        "knowledge_files_count": len(knowledge_manager.get_knowledge_files())
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """通常のチャット応答"""
    try:
        # ナレッジファイルを検索
        knowledge_files = intent_clarification_service._search_knowledge(message.message)
        
        # シンプルな応答を生成
        if knowledge_files:
            response = f"「{message.message}」について、以下のナレッジファイルが関連しています：\n\n"
            for file in knowledge_files[:3]:
                response += f"• {file['title']} ({file['category']})\n"
        else:
            response = f"「{message.message}」について、関連するナレッジファイルが見つかりませんでした。"
        
        return ChatResponse(
            response=response,
            knowledge_files=knowledge_files,
            chat_id=message.chat_id or "default"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/search")
async def search_knowledge(query: str):
    """ナレッジ検索エンドポイント"""
    try:
        results = intent_clarification_service._search_knowledge(query)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/files")
async def get_knowledge_files():
    """利用可能なナレッジファイル一覧"""
    files = knowledge_manager.get_knowledge_files()
    return {"files": files}

@app.post("/knowledge/refresh")
async def refresh_knowledge():
    """ナレッジファイルを強制的に再スキャン"""
    try:
        knowledge_manager.last_scan_time = None  # 強制再スキャン
        files = knowledge_manager.get_knowledge_files()
        return {"message": "Knowledge files refreshed", "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 意図特定機能のAPIエンドポイント
@app.post("/intent/start", response_model=IntentClarificationResponse)
async def start_intent_clarification(request: IntentClarificationRequest):
    """意図特定を開始"""
    try:
        is_ambiguous = intent_clarification_service.is_ambiguous(request.user_input)
        
        if is_ambiguous:
            question = intent_clarification_service.generate_question(request.user_input, 1)
            return IntentClarificationResponse(
                question=question,
                is_ambiguous=True
            )
        else:
            # 曖昧でない場合は直接検索
            knowledge_files = intent_clarification_service._search_knowledge(request.user_input)
            question = intent_clarification_service.generate_question(request.user_input, 1)
            return IntentClarificationResponse(
                question=question,
                is_ambiguous=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/question")
async def generate_next_question(request: IntentQuestionRequest):
    """次の質問を生成"""
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
    """意図特定を完了"""
    try:
        intent_result = intent_clarification_service.identify_intent(request.user_input, request.answers)
        return intent_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/search-by-intent")
async def search_knowledge_by_intent(intent_result: IntentResult):
    """意図に基づいてナレッジを検索"""
    try:
        return {
            "knowledge_files": intent_result.knowledge_files,
            "intent": intent_result.primary_intent,
            "keywords": intent_result.keywords,
            "explanation": f"あなたの意図「{intent_result.primary_intent}」に基づいて、関連するナレッジを表示しています。"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 