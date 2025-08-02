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
    conversation_history: List[Dict[str, str]] = []

# 対話状態管理のための新しいクラスを追加
class ConversationState(BaseModel):
    phase: str = "initial"  # "initial", "knowledge_shown", "deep_dive"
    shown_knowledge: List[str] = []  # 既に表示したファイルタイトル
    user_interests: Dict = {}  # ユーザの関心領域
    conversation_depth: int = 0  # 深掘り回数

# ChatResponseを更新して対話状態を含める
class ChatResponse(BaseModel):
    response: str
    knowledge_files: List[Dict]
    chat_id: str
    needs_clarification: bool = False
    continuation_message: str = ""  # 継続促進メッセージ
    conversation_state: ConversationState = ConversationState()

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
                    "prompt": prompt[:3000],  # プロンプト長をさらに拡張
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,    # 1024から2048に拡張
                        "num_thread": 4,   # 2から4に拡張
                        "temperature": 0.7  # 0.5から0.7に拡張（創造性向上）
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"LLM API error: {response.text}")
                return "LLMからの応答を取得できませんでした。"
                
        except requests.exceptions.ConnectionError as e:
            print(f"LLM接続エラー: {e}")
            return "LLMサービスに接続できませんでした。"
        except MemoryError as e:
            print(f"メモリエラー: {e}")
            return "メモリ不足のため処理を中断しました。"
        except Exception as e:
            print(f"LLM API error: {e}")
            return "LLMからの応答を取得できませんでした。"

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
    
    def needs_clarification(self, user_input: str, conversation_history: List[Dict[str, str]] = []) -> bool:
        """質問の掘り下げが必要かどうかを判断"""
        
        # 会話のターン数をチェック（3ターン以上なら強制的にナレッジ表示）
        user_turn_count = sum(1 for entry in conversation_history if entry.get('role') == 'user')
        if user_turn_count >= 2:  # 3ターン目（0,1,2）でナレッジ表示
            print(f"3ターン目に到達: {user_turn_count + 1}ターン目、ナレッジ表示に移行")
            return False
        
        # 具体的なキーワードが含まれている場合は即座にナレッジ表示
        specific_keywords = [
            "育成プログラム", "研修制度", "研修方法", "施策", "事例", "具体的な方法", "対策", "取り組み方法",
            "エンゲージメント向上", "離職防止施策", "管理職育成", "新人研修", "若手育成", "中堅研修", "部長育成",
            "組織風土改革", "女性活躍推進", "コーチング手法", "人材育成制度", "デジタル化", "ハラスメント対策"
        ]
        
        if any(keyword in user_input for keyword in specific_keywords):
            print(f"具体的キーワード検出: ナレッジ表示に移行")
            return False
        
        # 初回で業界が明確な場合は2ターン目でナレッジ表示
        industry_keywords = [
            "IT業界", "製造業界", "建設業界", "医療業界", "金融業界", "教育業界",
            "エネルギー業界", "通信業界", "運輸業界", "小売業界", "化学業界", "自動車業界", "食品業界"
        ]
        
        if user_turn_count == 0 and any(keyword in user_input for keyword in industry_keywords):
            print("初回で業界明確: 次ターンでナレッジ表示予定")
            return True  # 1回だけ掘り下げ
        
        # 会話履歴から文脈を構築
        context = ""
        if conversation_history:
            recent_history = conversation_history[-2:]  # 最新2件に短縮
            context_parts = []
            for entry in recent_history:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:50]}")
                elif entry.get('role') == 'assistant':
                    context_parts.append(f"AI: {entry.get('content', '')[:50]}")
            
            if context_parts:
                context = "会話履歴:\n" + "\n".join(context_parts) + "\n\n"
        
        # 簡潔な判断プロンプト
        prompt = f"""
{context}現在のユーザーの発言:「{user_input}」

この発言が十分に具体的でナレッジを提示できるかを判断してください。

ナレッジ提示できる場合（掘り下げ不要）:
- 業界が明確
- 何かしらの具体的なテーマや関心事がある
- 質問の意図が理解できる

掘り下げが必要な場合:
- 業界も具体的テーマも全く不明確
- 「教えて」だけで何について知りたいか全くわからない

必要なら「はい」、不要なら「いいえ」で回答してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt).strip().lower()
            # より厳密な判断：明確に「はい」と回答した場合のみ掘り下げ
            return response in ['はい', 'yes', '必要', 'true', '1']
        except Exception as e:
            print(f"掘り下げ判断エラー: {e}")
            # エラーの場合は安全のため掘り下げを提案
            return True
    
    def generate_clarification_question(self, user_input: str, conversation_history: List[Dict[str, str]] = []) -> str:
        """質問の掘り下げを行う簡潔な質問を生成"""
        # 会話履歴から文脈を構築
        context = ""
        if conversation_history:
            recent_history = conversation_history[-1:]  # 最新1件のみ
            context_parts = []
            for entry in recent_history:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:50]}")
            
            if context_parts:
                context = "前回: " + context_parts[0] + "\n\n"
        
        # 簡潔な質問生成プロンプト
        prompt = f"""
{context}現在の発言:「{user_input}」

次のターンでナレッジを提示できるよう、簡潔で的確な1つの質問をしてください。

質問例:
- 業界が不明 → 「どちらの業界でしょうか？」
- テーマが広い → 「具体的にはどのような点が気になりますか？」
- 対象が不明 → 「どのような立場の方についてでしょうか？」

重要: 短く、答えやすい質問を1つだけ。日本語のみ。
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
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> str:
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
                return "理解しました。関連するナレッジファイルを検索します。"
            
            return response
        except Exception as e:
            # フォールバック（制約最小化）
            question = f"「{user_input}」について、より詳しい情報をお聞かせください。"
            return question
    
    def search_obsidian_with_llm(self, user_needs: str, conversation_history: List[Dict[str, str]] = []) -> List[Dict]:
        files = self.knowledge_manager.get_knowledge_files()
        
        if not files:
            print("ナレッジファイルが見つかりません")
            return []
        
        # ファイルリストを適度に制限
        limited_files = files[:50]
        
        # 会話履歴から文脈を構築
        context = ""
        if conversation_history:
            recent = conversation_history[-3:]  # 最新3件で文脈をより詳細に
            context_parts = []
            for entry in recent:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:80]}")
                elif entry.get('role') == 'assistant':
                    context_parts.append(f"AI: {entry.get('content', '')[:80]}")
            
            if context_parts:
                context = "会話履歴:\n" + "\n".join(context_parts) + "\n\n"
        
        # 文脈重視のプロンプト
        prompt = f"""
{context}ユーザーの質問:「{user_needs}」

以下のファイルから関連性の高いものを3-5個選択してください：

{chr(10).join([f"- {file['title']}" for file in limited_files[:25]])}

選択のポイント:
- 会話の文脈に沿ったファイル
- 質問に直接関連するファイル
- 業界が特定されている場合はその業界を優先

重要: 日本語のみ。JSON形式で回答してください。

JSON形式:
{{"files": [{{"title": "ファイル名", "relevance": "高"}}]}}
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            print(f"LLM応答: {response}")
            
            # LLMがエラーメッセージを返した場合はフォールバック
            if any(error_keyword in response for error_keyword in ['エラー', '取得できません', '通信でエラー']):
                print("LLMエラー検出、フォールバック検索を実行")
                fallback_results = self._fallback_search(user_needs, files, conversation_history)
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
                    fallback_results = self._fallback_search(user_needs, files, conversation_history)
                    if fallback_results:
                        print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
                    return fallback_results
            else:
                # JSON形式で応答しなかった場合のフォールバック
                print("LLMがJSON形式で応答しませんでした、フォールバック検索を実行")
                fallback_results = self._fallback_search(user_needs, files, conversation_history)
                if fallback_results:
                    print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
                return fallback_results
        except Exception as e:
            print(f"LLM検索エラー: {e}")
            # エラー時のフォールバック
            fallback_results = self._fallback_search(user_needs, files, conversation_history)
            if fallback_results:
                print(f"フォールバック検索で{len(fallback_results)}個のファイルを発見")
            return fallback_results
    
    def _fallback_search(self, user_needs: str, files: List[Dict], conversation_history: List[Dict[str, str]] = []) -> List[Dict]:
        """フォールバック検索: LLMによる柔軟な検索"""
        print(f"フォールバック検索実行: {user_needs}")
        
        # ファイルリストを適度に制限
        limited_files = files[:25]
        
        # 会話履歴から文脈を構築
        context = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # 最新3件
            context_parts = []
            for entry in recent_history:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:80]}")
                elif entry.get('role') == 'assistant':
                    context_parts.append(f"AI: {entry.get('content', '')[:80]}")
            
            if context_parts:
                context = "会話履歴:\n" + "\n".join(context_parts) + "\n\n"
        
        # 文脈重視のフォールバック検索プロンプト
        prompt = f"""
あなたは文脈を理解して自然な対話を行うアシスタントです。

{context}現在のユーザーの発言:「{user_needs}」

以下のファイルから関連性の高いものを3つ選択してください：

{chr(10).join([f"- {file['title']}" for file in limited_files])}

重要な文脈継続の原則:
- 会話の流れを重視し、これまでの話題との関連性を最優先する
- 業界・分野・テーマが既に決まっている場合は、その文脈を継続する
- ユーザーの真の関心事を理解し、会話の自然な流れに沿ったファイルを選ぶ
- 「一般的に」「全般的に」といった表現は、現在の文脈内での全般を意味する

重要: あなたは日本語で回答するアシスタントです。英語や中国語は一切使用しないでください。

JSON形式で回答:
{{"files": [{{"title": "ファイル名", "relevance": "高"}}]}}
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

# 段階的ナレッジ提示システムの核クラス
class StepwiseKnowledgeService:
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
        self.llm_service = LLMService()
        self.conversation_states = {}  # chat_id -> ConversationState
    
    def should_show_knowledge(self, user_input: str, conversation_history: List[Dict[str, str]] = []) -> bool:
        """初回ナレッジ提示の判断（緩めの基準）"""
        
        # 非常にシンプルな判断基準
        # 1. 何らかの業界や分野が特定できる
        # 2. 具体的なテーマが含まれている
        # 3. 質問の意図が理解できる
        
        keywords_industry = [
            "IT業界", "製造業界", "建設業界", "医療業界", "金融業界", "教育業界",
            "エネルギー業界", "通信業界", "運輸業界", "小売業界", "化学業界", "自動車業界", "食品業界"
        ]
        
        keywords_theme = [
            "人材育成", "エンゲージメント", "離職", "管理職", "新人", "若手", "中堅", "研修",
            "組織風土", "女性活躍", "コーチング", "ハラスメント", "デジタル化", "DX"
        ]
        
        # 業界 OR テーマが含まれていれば提示
        has_industry = any(keyword in user_input for keyword in keywords_industry)
        has_theme = any(keyword in user_input for keyword in keywords_theme)
        
        return has_industry or has_theme
    
    def generate_knowledge_response(self, user_input: str, conversation_history: List[Dict[str, str]], 
                                  chat_id: str, is_continuation: bool = False) -> ChatResponse:
        """段階的ナレッジ提示のメイン処理"""
        
        # 対話状態の取得/初期化
        if chat_id not in self.conversation_states:
            self.conversation_states[chat_id] = ConversationState()
        
        state = self.conversation_states[chat_id]
        
        # ナレッジ検索
        knowledge_files = self._search_knowledge(user_input, conversation_history, state)
        
        # 継続促進メッセージの生成
        continuation_msg = self._generate_continuation_message(knowledge_files, is_continuation)
        
        # 対話状態の更新
        state.conversation_depth += 1
        state.phase = "knowledge_shown" if not is_continuation else "deep_dive"
        state.shown_knowledge.extend([f['title'] for f in knowledge_files])
        
        return ChatResponse(
            response="下記のようなナレッジが見つかりました：",
            knowledge_files=knowledge_files,
            chat_id=chat_id,
            needs_clarification=False,
            continuation_message=continuation_msg,
            conversation_state=state
        )
    
    def _search_knowledge(self, user_input: str, conversation_history: List[Dict[str, str]], 
                         state: ConversationState) -> List[Dict]:
        """段階的検索ロジック"""
        if state.phase == "initial":
            # 初回は広めの検索
            return self._broad_search(user_input, conversation_history)
        else:
            # 継続は絞り込んだ検索
            return self._focused_search(user_input, conversation_history, state.shown_knowledge)
    
    def _broad_search(self, user_input: str, conversation_history: List[Dict[str, str]]) -> List[Dict]:
        """広めの初回検索"""
        # 文脈を構築
        context = ""
        if conversation_history:
            recent_history = conversation_history[-2:]
            context_parts = []
            for entry in recent_history:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:80]}")
                elif entry.get('role') == 'assistant':
                    context_parts.append(f"AI: {entry.get('content', '')[:80]}")
            
            if context_parts:
                context = "会話履歴:\n" + "\n".join(context_parts) + "\n\n"
        
        # 利用可能なファイルリスト
        all_files = self.knowledge_manager.get_knowledge_files()
        limited_files = all_files[:30]  # メモリ制約のため制限
        
        prompt = f"""
{context}ユーザーの質問:「{user_input}」

以下のファイルから関連性の高いものを3-4個選択してください：

{chr(10).join([f"- {file['title']}" for file in limited_files])}

重要: 日本語のみ。JSON形式で回答してください。

JSON形式:
{{"files": [{{"title": "ファイル名", "relevance": "高"}}]}}
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            return self._parse_llm_file_selection(response, all_files)
        except Exception as e:
            print(f"LLM検索エラー: {e}")
            return self._fallback_search(user_input)[:3]
    
    def _focused_search(self, user_input: str, conversation_history: List[Dict[str, str]], 
                       shown_files: List[str]) -> List[Dict]:
        """絞り込んだ継続検索"""
        all_files = self.knowledge_manager.get_knowledge_files()
        
        # 既に表示したファイルを除外
        available_files = [f for f in all_files if f['title'] not in shown_files]
        limited_files = available_files[:20]
        
        # 文脈を構築
        context = ""
        if conversation_history:
            recent_history = conversation_history[-2:]
            context_parts = []
            for entry in recent_history:
                if entry.get('role') == 'user':
                    context_parts.append(f"ユーザー: {entry.get('content', '')[:80]}")
            
            if context_parts:
                context = "前回の質問: " + context_parts[-1] + "\n\n"
        
        prompt = f"""
{context}追加質問:「{user_input}」

既に表示済み: {', '.join(shown_files[:3])}

以下の未表示ファイルから、さらに詳しい情報として関連性の高いものを2-3個選択してください：

{chr(10).join([f"- {file['title']}" for file in limited_files])}

重要: より具体的・詳細な情報を優先。JSON形式で回答してください。

JSON形式:
{{"files": [{{"title": "ファイル名", "relevance": "高"}}]}}
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            return self._parse_llm_file_selection(response, available_files)
        except Exception as e:
            print(f"継続検索エラー: {e}")
            return available_files[:2]
    
    def _generate_continuation_message(self, knowledge_files: List[Dict], is_continuation: bool) -> str:
        """継続促進メッセージの生成"""
        if not knowledge_files:
            return ""
        
        if is_continuation:
            return "💬 他に気になる点があれば、お話しください。"
        else:
            # 初回表示時のメッセージ
            examples = []
            for file in knowledge_files[:2]:  # 最初の2つから例を生成
                title = file.get('title', '')
                if 'IT業界' in title:
                    examples.append('「IT業界の具体的な事例は？」')
                elif '製造業界' in title:
                    examples.append('「製造業での実際の効果は？」')
                elif '育成' in title:
                    examples.append('「研修期間や方法について詳しく知りたい」')
                elif 'エンゲージメント' in title:
                    examples.append('「エンゲージメント測定方法は？」')
            
            if not examples:
                examples = ['「具体的な事例は？」', '「実際の効果や数値データは？」']
            
            example_text = '、'.join(examples[:2])
            return f"💬 さらに詳細にテーマを絞りたい場合は、お話しください。\n例：{example_text}"
    
    def _parse_llm_file_selection(self, llm_response: str, available_files: List[Dict]) -> List[Dict]:
        """LLMレスポンスからファイル選択をパース"""
        try:
            # JSONを抽出
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                selected_files = []
                if 'files' in data and isinstance(data['files'], list):
                    for item in data['files']:
                        if isinstance(item, dict) and 'title' in item:
                            title = item['title']
                            # ファイルを検索
                            for file in available_files:
                                if file['title'] == title:
                                    selected_files.append(file)
                                    break
                
                return selected_files[:5]  # 最大5件
        except Exception as e:
            print(f"LLM応答パースエラー: {e}")
        
        # フォールバック
        return available_files[:3]
    
    def _fallback_search(self, user_input: str) -> List[Dict]:
        """フォールバック検索（簡単な文字列マッチング）"""
        all_files = self.knowledge_manager.get_knowledge_files()
        matched_files = []
        
        # 簡単なキーワードマッチング
        for file in all_files:
            title = file.get('title', '').lower()
            if any(keyword.lower() in title for keyword in user_input.split()):
                matched_files.append(file)
        
        return matched_files[:3] if matched_files else all_files[:3]

# グローバル変数としてサービスを定義
print("KnowledgeManager初期化開始...")
knowledge_manager = KnowledgeManager("../KNOWLEDGE")
print(f"KnowledgeManager初期化完了: {len(knowledge_manager.get_knowledge_files())}個のファイルを読み込み")

print("IntentClarificationService初期化開始...")
intent_clarification_service = IntentClarificationService(knowledge_manager)
print("IntentClarificationService初期化完了")

# 新しい段階的ナレッジ提示サービスを初期化
print("StepwiseKnowledgeService初期化開始...")
stepwise_service = StepwiseKnowledgeService(knowledge_manager)
print("StepwiseKnowledgeService初期化完了")

# APIエンドポイント
# メインのチャットエンドポイント
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        user_input = message.message.strip()
        chat_id = message.chat_id or "default"
        conversation_history = message.conversation_history or []
        
        # 段階的ナレッジ提示の判断
        if stepwise_service.should_show_knowledge(user_input, conversation_history):
            # 継続対話かどうかを判断
            is_continuation = chat_id in stepwise_service.conversation_states
            
            # ナレッジを提示
            return stepwise_service.generate_knowledge_response(
                user_input, conversation_history, chat_id, is_continuation
            )
        else:
            # 質問の掘り下げが必要
            if intent_clarification_service.needs_clarification(user_input, conversation_history):
                clarification_question = intent_clarification_service.generate_clarification_question(
                    user_input, conversation_history
                )
                
                return ChatResponse(
                    response=clarification_question,
                    knowledge_files=[],
                    chat_id=chat_id,
                    needs_clarification=True,
                    continuation_message="",
                    conversation_state=ConversationState()
                )
            else:
                # 直接ナレッジ検索
                return stepwise_service.generate_knowledge_response(
                    user_input, conversation_history, chat_id, False
                )
                
    except Exception as e:
        print(f"チャットエラー: {e}")
        return ChatResponse(
            response="申し訳ございませんが、エラーが発生しました。もう一度お試しください。",
            knowledge_files=[],
            chat_id=chat_id or "default",
            needs_clarification=False,
            continuation_message="",
            conversation_state=ConversationState()
        )

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

def _generate_response(user_message: str, knowledge_files: List[Dict]) -> str:
    """LLMを使って応答を生成"""
    print(f"回答生成: 現在の質問: {user_message}")
    
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    # シンプルなメッセージのみ返す（LLMでの詳細回答生成を廃止）
    return "下記のようなナレッジが見つかりました："

if __name__ == "__main__":
    import uvicorn
    print("FastAPIサーバー起動中...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 