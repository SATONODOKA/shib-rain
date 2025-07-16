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

class KnowledgeManager:
    """ナレッジファイルを動的に管理するクラス"""
    
    def __init__(self, knowledge_base_path: str = "../KNOWLEDGE"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_files = []
        self.last_scan_time = None
        self.scan_interval = 60  # 60秒ごとにスキャン
    
    def scan_knowledge_files(self) -> List[Dict]:
        """ナレッジファイルをスキャンして構造を取得"""
        current_time = datetime.now()
        
        # スキャン間隔をチェック
        if (self.last_scan_time and 
            (current_time - self.last_scan_time).seconds < self.scan_interval):
            return self.knowledge_files
        
        self.last_scan_time = current_time
        knowledge_files = []
        
        if not self.knowledge_base_path.exists():
            print(f"Warning: Knowledge base path {self.knowledge_base_path} does not exist")
            return knowledge_files
        
        # 業界別フォルダをスキャン
        industry_path = self.knowledge_base_path / "業界別"
        if industry_path.exists():
            for file_path in industry_path.glob("*.md"):
                knowledge_file = self._parse_knowledge_file(file_path, "業界別")
                if knowledge_file:
                    knowledge_files.append(knowledge_file)
        
        # 能力開発テーマ別フォルダをスキャン
        theme_path = self.knowledge_base_path / "能力開発テーマ"
        if theme_path.exists():
            for file_path in theme_path.glob("*.md"):
                knowledge_file = self._parse_knowledge_file(file_path, "能力開発テーマ")
                if knowledge_file:
                    knowledge_files.append(knowledge_file)
        
        self.knowledge_files = knowledge_files
        print(f"Scanned {len(knowledge_files)} knowledge files")
        return knowledge_files
    
    def _parse_knowledge_file(self, file_path: Path, category: str) -> Optional[Dict]:
        """ナレッジファイルを解析して構造を取得"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # ファイル名から情報を抽出
            filename = file_path.stem
            relative_path = str(file_path.relative_to(self.knowledge_base_path))
            
            # タイトルを抽出（最初の#行）
            title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else filename
            
            # タグを抽出
            tags = []
            tag_match = re.search(r'\*\*タグ\*\*: (.+)', content)
            if tag_match:
                tag_text = tag_match.group(1)
                tags = [tag.strip('#') for tag in tag_text.split() if tag.startswith('#')]
            
            # 説明を抽出（概要セクションから）
            description = ""
            description_match = re.search(r'## 📋 概要\s*\n\s*(.+?)(?=\n##|\n---|$)', content, re.DOTALL)
            if description_match:
                description = description_match.group(1).strip()
            else:
                # 概要がない場合は最初の段落を使用
                first_para_match = re.search(r'^(.+?)(?=\n\n|\n##|\n---|$)', content, re.DOTALL)
                if first_para_match:
                    description = first_para_match.group(1).strip()
            
            # キーワードを生成（タイトル、タグ、説明から）
            keywords = set()
            keywords.update(tags)
            
            # タイトルからキーワードを抽出
            title_keywords = re.findall(r'[^\s_]+', title)
            keywords.update(title_keywords)
            
            # 説明からキーワードを抽出
            desc_keywords = re.findall(r'[^\s、。]+', description)
            keywords.update(desc_keywords)
            
            # カテゴリ固有のキーワードを追加
            if category == "業界別":
                if "業界一般" in filename:
                    keywords.add("業界一般")
                else:
                    keywords.add("バリューチェーン")
            
            return {
                "path": relative_path,
                "title": title,
                "category": category,
                "keywords": list(keywords),
                "description": description,
                "file_size": len(content),
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def get_knowledge_files(self) -> List[Dict]:
        """最新のナレッジファイル一覧を取得"""
        return self.scan_knowledge_files()

class LLMService:
    """ローカルLLMとの通信を管理するクラス"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "qwen2.5:7b"
        self.is_available = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Ollamaが利用可能かチェック"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, user_message: str, context: str = "") -> str:
        """LLMを使用して応答を生成"""
        if not self.is_available:
            return self._fallback_response(user_message)
        
        try:
            # プロンプトの構築
            prompt = self._build_prompt(user_message, context)
            
            # Ollama APIにリクエスト
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return self._fallback_response(user_message)
                
        except Exception as e:
            print(f"LLM API error: {e}")
            return self._fallback_response(user_message)
    
    def _build_prompt(self, user_message: str, context: str) -> str:
        """プロンプトを構築"""
        return f"""あなたは業界知識と能力開発の専門家です。ユーザーの質問に対して、以下のナレッジファイルから関連性の高い情報を提供してください。

利用可能なナレッジファイル:
{context}

ユーザーの質問: {user_message}

回答の際は以下の点に注意してください:
1. 質問の意図を理解し、適切なナレッジファイルを参照する
2. 具体的で実用的なアドバイスを提供する
3. 関連するナレッジファイルの情報を活用する
4. 建設的で前向きな回答をする
5. 必ず日本語で回答する

回答:"""
    
    def _fallback_response(self, user_message: str) -> str:
        """LLMが利用できない場合のフォールバック応答"""
        return f"申し訳ございませんが、現在ローカルLLMが利用できません。質問「{user_message}」について、ナレッジファイルから関連情報を検索してお答えします。"

class KnowledgeService:
    """ナレッジ検索を管理するクラス"""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
    
    def search_knowledge(self, query: str, llm_enhanced: bool = True) -> List[Dict]:
        """ナレッジファイルを検索"""
        # 最新のナレッジファイルを取得
        knowledge_files = self.knowledge_manager.get_knowledge_files()
        
        if llm_enhanced:
            # LLMを使用してクエリを改善
            enhanced_query = self._enhance_query_with_llm(query)
            return self._search_with_score(enhanced_query, knowledge_files)
        else:
            return self._search_with_score(query, knowledge_files)
    
    def _enhance_query_with_llm(self, query: str) -> str:
        """LLMを使用してクエリを改善"""
        llm_service = LLMService()
        
        enhancement_prompt = f"""以下の質問を、ナレッジ検索に適したキーワードに変換してください。必ず日本語で回答してください。

質問: {query}

利用可能なナレッジカテゴリ:
- 業界: 建設業界、製造業、IT業界、金融業界、サービス業、小売業、医療業界、教育業界、運輸業界
- テーマ: 組織風土改革、女性活躍、働き方改革、採用戦略、人材育成、評価制度、コミュニケーション、意思決定、プロジェクト管理、リーダーシップ、コーチング、若手の自立、エンゲージメント、コンサル
- バリューチェーン: 現場職人、現場監督、設計者、デベロッパー、コンサルタント、管理職、経営層

検索キーワード（カンマ区切り）:"""
        
        try:
            enhanced = llm_service.generate_response(query, enhancement_prompt)
            # キーワードを抽出
            keywords = [kw.strip() for kw in enhanced.split(',') if kw.strip()]
            return ' '.join(keywords) if keywords else query
        except:
            return query
    
    def _search_with_score(self, query: str, knowledge_files: List[Dict]) -> List[Dict]:
        """スコアリング付きでナレッジファイルを検索"""
        results = []
        query_lower = query.lower()
        
        for file in knowledge_files:
            score = 0
            
            # タイトルでのマッチング
            if file["title"].lower() in query_lower:
                score += 10
            
            # キーワードでのマッチング
            for keyword in file["keywords"]:
                if keyword.lower() in query_lower:
                    score += 5
                # 部分一致
                if keyword.lower() in query_lower or query_lower in keyword.lower():
                    score += 3
            
            # 説明でのマッチング
            if file["description"].lower() in query_lower:
                score += 3
            
            # カテゴリでのマッチング
            if file["category"].lower() in query_lower:
                score += 2
            
            if score > 0:
                results.append({
                    **file,
                    "score": score
                })
        
        # スコアでソート
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# サービスインスタンス
knowledge_manager = KnowledgeManager()
llm_service = LLMService()
knowledge_service = KnowledgeService(knowledge_manager)

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "ナレッジチャットAPI", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    knowledge_files = knowledge_manager.get_knowledge_files()
    return {
        "status": "healthy",
        "llm_available": llm_service.is_available,
        "knowledge_files_count": len(knowledge_files),
        "last_scan_time": knowledge_manager.last_scan_time.isoformat() if knowledge_manager.last_scan_time else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """チャットエンドポイント"""
    try:
        # ナレッジ検索（通常検索）
        knowledge_results = knowledge_service.search_knowledge(message.message, llm_enhanced=False)
        
        # LLMを使用して応答を生成
        context = "\n".join([
            f"- {file['title']}: {file['description']}"
            for file in knowledge_results[:3]  # 上位3件をコンテキストとして使用
        ])
        
        llm_response = llm_service.generate_response(message.message, context)
        
        # チャットIDの生成
        chat_id = message.chat_id or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ChatResponse(
            response=llm_response,
            knowledge_files=knowledge_results,
            chat_id=chat_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/search")
async def search_knowledge(query: str):
    """ナレッジ検索エンドポイント"""
    try:
        results = knowledge_service.search_knowledge(query, llm_enhanced=False)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 