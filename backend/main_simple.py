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

# LLMサービス
class LLMService:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "qwen2.5:7b"
        self.is_available = self._check_ollama_availability()
    
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

# ナレッジ管理
class KnowledgeManager:
    def __init__(self, knowledge_base_path: str = "../KNOWLEDGE"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_files = []
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
            
            # キーワードを抽出
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
        """キーワードを抽出"""
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

# 意図特定サービス
class IntentClarificationService:
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge_manager = knowledge_manager
        self.llm_service = LLMService()
    
    def is_ambiguous(self, user_input: str) -> bool:
        """入力が曖昧かどうかを判定"""
        ambiguous_words = ["知りたい", "教えて", "について", "どう", "何か"]
        return any(word in user_input for word in ambiguous_words)
    
    def has_sufficient_information(self, user_input: str, answers: List[str]) -> bool:
        """十分な情報が得られたかを判定"""
        if not answers:
            return False
        
        # 回答の長さと内容をチェック
        total_length = sum(len(answer) for answer in answers)
        if total_length < 10:  # 合計10文字未満は不十分
            return False
        
        # 具体的なキーワードが含まれているかをチェック
        specific_keywords = [
            "建設業界", "パワハラ", "研修", "事例", "対策", "防止", "管理職", 
            "現場", "職人", "女性", "育成", "組織", "風土", "改革", "採用",
            "新卒", "働き方", "人材", "技能", "コミュニケーション", "安全"
        ]
        
        combined_text = f"{user_input} {' '.join(answers)}"
        found_keywords = [kw for kw in specific_keywords if kw in combined_text]
        
        # 2つ以上の具体的なキーワードがあれば十分
        if len(found_keywords) >= 2:
            return True
        
        # 回答が具体的で詳細な場合
        detailed_indicators = ["具体的", "詳細", "実例", "実際", "具体的な", "具体的に"]
        if any(indicator in combined_text for indicator in detailed_indicators):
            return True
        
        # 回答の長さが十分で、具体的な内容が含まれている場合
        if total_length >= 30 and any(len(answer) >= 15 for answer in answers):
            return True
        
        return False
    
    def generate_question(self, user_input: str, question_number: int, answers: List[str] = None) -> IntentQuestion:
        """LLMを使用して文脈を考慮した質問を生成"""
        if answers is None:
            answers = []
        
        # 十分な情報が得られた場合は早期完了を示す
        if self.has_sufficient_information(user_input, answers):
            return IntentQuestion(
                question="十分な情報が得られました。回答を生成します。",
                question_number=question_number,
                max_questions=3,
                is_complete=True
            )
        
        if self.llm_service.is_available:
            return self._generate_question_with_llm(user_input, question_number, answers)
        else:
            return self._generate_fallback_question(user_input, question_number, answers)
    
    def _generate_question_with_llm(self, user_input: str, question_number: int, answers: List[str]) -> IntentQuestion:
        """LLMを使用して質問を生成"""
        conversation_context = f"ユーザーの最初の質問: {user_input}"
        if answers:
            conversation_context += f"\nこれまでの回答: {' '.join(answers)}"
        
        prompt = f"""
あなたは人材開発・組織開発のコンサルタントです。ユーザーが知りたい具体的な情報を特定するための質問を生成してください。

ユーザーの最初の質問: {user_input}
これまでの回答: {', '.join(answers) if answers else 'なし'}

質問番号: {question_number}/3

以下の条件に従って質問を生成してください：
1. ユーザーが知りたい具体的な内容を掘り下げる質問をする
2. 「どのような情報をお探しですか？」という方向性で質問する
3. 必ず日本語で回答してください
4. 質問のみを返し、余計な文章は含めないでください
5. 質問は簡潔に、1文で終わるようにしてください

質問例：
「建設業界についてですね。どのような情報をお探しですか？」
「パワハラ防止について、どのような具体的な情報をお探しですか？」
「人材育成の課題について、どのような具体的な課題に興味がありますか？」

ユーザーの質問「{user_input}」に基づいて、ユーザーが知りたい具体的な内容を掘り下げる質問を1つだけ生成してください。
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            question = response.strip()
            
            # 応答が空またはエラーの場合のフォールバック
            if not question or "申し訳" in question or "エラー" in question:
                return self._generate_fallback_question(user_input, question_number, answers)
            
            return IntentQuestion(
                question=question,
                question_number=question_number,
                max_questions=3
            )
        except Exception as e:
            print(f"LLM質問生成エラー: {e}")
            return self._generate_fallback_question(user_input, question_number, answers)
    
    def _generate_fallback_question(self, user_input: str, question_number: int, answers: List[str]) -> IntentQuestion:
        """フォールバック用の質問生成"""
        context = f"{user_input}"
        if answers:
            context += f" {' '.join(answers)}"
        
        if question_number == 1:
            if "建設" in user_input:
                question = f"「{user_input}」について、どのようなテーマに興味がありますか？市場動向、人材育成の課題、成功事例など、どのような情報をお探しですか？"
            elif "パワハラ" in user_input or "ハラスメント" in user_input:
                question = f"「{user_input}」について、どのような具体的な情報をお探しですか？研修内容、事例、対策方法、法規制など"
            elif "人材" in user_input or "育成" in user_input:
                question = f"「{user_input}」について、どのような具体的な課題に興味がありますか？技能者不足、世代間ギャップ、研修制度など"
            else:
                question = f"「{user_input}」について、どのような情報をお探しですか？"
        elif question_number == 2:
            if "パワハラ" in context or "ハラスメント" in context:
                question = "パワハラ防止について、どのような具体的な情報をお探しですか？研修内容、事例、対策方法など"
            elif "建設" in context:
                question = "建設業界のどのような側面について知りたいですか？現場環境、管理職向け、新入社員向けなど"
            elif "人材" in context or "育成" in context:
                question = "人材育成について、どのような具体的な課題に興味がありますか？技能者不足、世代間ギャップ、研修制度など"
            else:
                question = "どのような立場の方向けの情報をお探しですか？"
        else:
            question = "どのような段階の情報をお探しですか？導入段階、運用段階、改善段階など"
        
        return IntentQuestion(
            question=question,
            question_number=question_number,
            max_questions=3
        )
    
    def identify_intent(self, user_input: str, answers: List[str]) -> IntentResult:
        """LLMを使用して意図を特定"""
        combined_input = f"{user_input} {' '.join(answers)}"
        
        if self.llm_service.is_available:
            return self._identify_intent_with_llm(user_input, answers)
        else:
            return self._identify_intent_fallback(combined_input)
    
    def _identify_intent_with_llm(self, user_input: str, answers: List[str]) -> IntentResult:
        """LLMを使用して意図を特定"""
        conversation_context = f"ユーザーの最初の質問: {user_input}"
        if answers:
            conversation_context += f"\nこれまでの回答: {' '.join(answers)}"
        
        prompt = f"""
あなたは人材開発・組織開発の専門家です。ユーザーの質問から、求めている情報の意図を分析してください。

ユーザーの最初の質問: {user_input}
これまでの回答: {', '.join(answers) if answers else 'なし'}

以下のJSON形式で回答してください：
{{
    "primary_intent": "主要な意図（業界動向、事例検索、提案・コンサルティング、課題解決、実践・運用のいずれか）",
    "keywords": ["具体的なキーワード1", "具体的なキーワード2", "具体的なキーワード3"],
    "confidence": 0.8
}}

キーワードは具体的で、ナレッジファイルの検索に使えるような詳細な単語を抽出してください。
例：「建設業界」→「建設業界」「現場職人」「技能者不足」
例：「パワハラ」→「パワハラ」「ハラスメント防止」「研修制度」

必ず日本語で回答し、JSONのみを返してください。説明や余計な文章は不要です。
"""
        
        try:
            response = self.llm_service.generate_response(prompt)
            print(f"LLM生応答: {response}")
            
            # JSONを抽出
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"抽出されたJSON文字列: {json_str}")
                
                data = json.loads(json_str)
                print(f"解析されたデータ: {data}")
                
                primary_intent = data.get("primary_intent", "一般情報")
                keywords = data.get("keywords", [])
                confidence = data.get("confidence", 0.5)
                
                # ナレッジファイルを検索
                combined_input = f"{user_input} {' '.join(answers)}"
                knowledge_files = self._search_knowledge(combined_input)
                
                result = IntentResult(
                    primary_intent=primary_intent,
                    keywords=keywords,
                    knowledge_files=knowledge_files
                )
                
                print(f"最終結果: primary_intent='{result.primary_intent}' keywords={result.keywords} knowledge_files={len(result.knowledge_files)}件")
                return result
            else:
                print("LLM意図特定エラー: JSONが見つかりません")
                return self._identify_intent_fallback(f"{user_input} {' '.join(answers)}")
                
        except Exception as e:
            print(f"LLM意図特定エラー: {e}")
            return self._identify_intent_fallback(f"{user_input} {' '.join(answers)}")
    
    def _identify_intent_fallback(self, combined_input: str) -> IntentResult:
        """フォールバック用の意図特定"""
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
        if "パワハラ" in text or "ハラスメント" in text or "撲滅" in text:
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
        if "パワハラ" in text or "ハラスメント" in text or "撲滅" in text:
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
        
        # クエリを単語に分割（日本語と英語の両方に対応）
        query_words = re.findall(r'[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', query.lower())
        
        for file in files:
            score = 0
            
            # タイトルでマッチング（完全一致と部分一致）
            title_lower = file["title"].lower()
            for word in query_words:
                if word in title_lower:
                    score += 5
                elif any(char in title_lower for char in word):
                    score += 2
            
            # キーワードでマッチング
            for keyword in file["keywords"]:
                keyword_lower = keyword.lower()
                for word in query_words:
                    if word in keyword_lower:
                        score += 4
                    elif any(char in keyword_lower for char in word):
                        score += 1
            
            # 説明文でマッチング
            desc_lower = file["description"].lower()
            for word in query_words:
                if word in desc_lower:
                    score += 2
                elif any(char in desc_lower for char in word):
                    score += 0.5
            
            # カテゴリでマッチング
            category_lower = file.get("category", "").lower()
            for word in query_words:
                if word in category_lower:
                    score += 3
            
            if score > 0:
                file["score"] = score
                results.append(file)
        
        # スコアでソート
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:5]  # 上位5件を返す

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
    """LLMを使用した自然なチャット応答"""
    try:
        # ナレッジファイルを検索
        knowledge_files = intent_clarification_service._search_knowledge(message.message)
        
        # LLMを使用して自然な応答を生成
        if intent_clarification_service.llm_service.is_available:
            response = await _generate_chat_response_with_llm(message.message, knowledge_files)
        else:
            response = _generate_chat_response_fallback(message.message, knowledge_files)
        
        return ChatResponse(
            response=response,
            knowledge_files=knowledge_files,
            chat_id=message.chat_id or "default"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_chat_response_with_llm(user_message: str, knowledge_files: List[Dict]) -> str:
    """ナレッジファイルの内容を基に回答を生成"""
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    # ナレッジファイルの内容を読み込んで回答を生成
    response_parts = []
    response_parts.append(f"「{user_message}」について、以下のナレッジファイルの情報をお答えします：\n")
    
    for file in knowledge_files[:3]:  # 最大3件まで
        try:
            rel_path = file['path'].strip()
            print(f"file['path']: {rel_path}")
            # 既にKNOWLEDGE/で始まる場合はそのまま
            if rel_path.startswith("KNOWLEDGE/"):
                file_path = Path("..") / rel_path
            else:
                file_path = Path("../KNOWLEDGE") / rel_path
            print(f"最終的なfile_path: {file_path.resolve()}")
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                print(f"ファイル読み込み成功: {file_path}, サイズ: {len(content)}")
                # ファイルの内容を簡潔に要約
                lines = content.split('\n')
                summary = []
                # 最初の数行から重要な情報を抽出
                for i, line in enumerate(lines[:20]):  # 最初の20行のみ処理
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 重要な情報を含む行を抽出
                    if any(keyword in line for keyword in ['概要', '現状', '課題', '施策', '事例', '成果', '定義', '特徴']):
                        summary.append(f"• {line}")
                        if len(summary) >= 5:  # 最大5行まで
                            break
                response_parts.append(f"【{file['title']}】")
                if summary:
                    response_parts.extend(summary)
                else:
                    # 要約できない場合は、ファイルの説明を使用
                    description = file.get('description', '詳細な情報が含まれています')
                    if len(description) > 100:
                        description = description[:100] + "..."
                    response_parts.append(f"• {description}")
                response_parts.append("")  # 空行
            else:
                print(f"ファイルが存在しません: {file_path}")
                response_parts.append(f"【{file['title']}】- ファイルが見つかりません")
        except Exception as e:
            print(f"ファイル読み込みエラー {file['path']}: {e}")
            response_parts.append(f"【{file['title']}】- ファイル読み込みエラー")
    
    result = "\n".join(response_parts)
    print(f"生成された応答の長さ: {len(result)}")
    return result

def _generate_chat_response_fallback(user_message: str, knowledge_files: List[Dict]) -> str:
    """フォールバック用のチャット応答（ナレッジファイルの内容を直接返す）"""
    if not knowledge_files:
        return f"「{user_message}」について、関連するナレッジファイルが見つかりませんでした。"
    
    # ナレッジファイルの内容を読み込んで回答を生成
    response_parts = []
    response_parts.append(f"「{user_message}」について、以下のナレッジファイルの情報をお答えします：\n")
    
    for file in knowledge_files[:3]:  # 最大3件まで
        try:
            rel_path = file['path'].strip()
            print(f"フォールバック: file['path']: {rel_path}")
            # 既にKNOWLEDGE/で始まる場合はそのまま
            if rel_path.startswith("KNOWLEDGE/"):
                file_path = Path("..") / rel_path
            else:
                file_path = Path("../KNOWLEDGE") / rel_path
            print(f"フォールバック: 最終的なfile_path: {file_path.resolve()}")
            
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                print(f"フォールバック: ファイル読み込み成功: {file_path}, サイズ: {len(content)}")
                
                # ファイルの内容を簡潔に要約
                lines = content.split('\n')
                summary = []
                
                # 最初の数行から重要な情報を抽出
                for i, line in enumerate(lines[:20]):  # 最初の20行のみ処理
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 重要な情報を含む行を抽出
                    if any(keyword in line for keyword in ['概要', '現状', '課題', '施策', '事例', '成果', '定義', '特徴']):
                        summary.append(f"• {line}")
                        if len(summary) >= 5:  # 最大5行まで
                            break
                
                response_parts.append(f"【{file['title']}】")
                if summary:
                    response_parts.extend(summary)
                else:
                    # 要約できない場合は、ファイルの説明を使用
                    description = file.get('description', '詳細な情報が含まれています')
                    if len(description) > 100:
                        description = description[:100] + "..."
                    response_parts.append(f"• {description}")
                response_parts.append("")  # 空行
            else:
                print(f"フォールバック: ファイルが存在しません: {file_path}")
                response_parts.append(f"【{file['title']}】- ファイルが見つかりません")
        except Exception as e:
            print(f"フォールバック: ファイル読み込みエラー {file['path']}: {e}")
            response_parts.append(f"【{file['title']}】- ファイル読み込みエラー")
    
    result = "\n".join(response_parts)
    print(f"フォールバック: 生成された応答の長さ: {len(result)}")
    return result

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
        # 意図を特定
        intent_result = intent_clarification_service.identify_intent(request.user_input, request.answers)
        
        # 特定されたキーワードを使ってナレッジを検索
        combined_query = f"{request.user_input} {' '.join(request.answers)} {' '.join(intent_result.keywords)}"
        knowledge_files = intent_clarification_service._search_knowledge(combined_query)
        
        # 結果を更新
        intent_result.knowledge_files = knowledge_files
        
        return intent_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/complete-with-response", response_model=ChatResponse)
async def complete_intent_with_response(request: IntentCompleteRequest):
    """意図特定を完了して直接回答を生成"""
    try:
        # 意図を特定
        intent_result = intent_clarification_service.identify_intent(request.user_input, request.answers)
        
        # 特定されたキーワードを使ってナレッジを検索
        combined_query = f"{request.user_input} {' '.join(request.answers)} {' '.join(intent_result.keywords)}"
        knowledge_files = intent_clarification_service._search_knowledge(combined_query)
        
        # 回答を生成
        if knowledge_files:
            response = await _generate_chat_response_with_llm(combined_query, knowledge_files)
        else:
            response = _generate_chat_response_fallback(combined_query, knowledge_files)
        
        return ChatResponse(
            response=response,
            knowledge_files=knowledge_files,
            chat_id=request.user_input[:10]  # 簡易的なchat_id
        )
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

# グローバル変数としてサービスを定義
knowledge_manager = KnowledgeManager()
intent_clarification_service = IntentClarificationService(knowledge_manager)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 