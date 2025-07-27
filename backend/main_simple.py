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
    def __init__(self, knowledge_base_path: str = None):
        if knowledge_base_path is None:
            # バックエンドディレクトリから実行されることを想定
            self.knowledge_base_path = Path("../KNOWLEDGE")
            # もし存在しない場合は、現在のディレクトリからの相対パスを試す
            if not self.knowledge_base_path.exists():
                self.knowledge_base_path = Path("KNOWLEDGE")
        else:
            self.knowledge_base_path = Path(knowledge_base_path)
        
        self.knowledge_files = []
        self.scan_knowledge_files()
    
    def scan_knowledge_files(self) -> List[Dict]:
        """ナレッジファイルをスキャン"""
        print(f"ナレッジベースパス: {self.knowledge_base_path}")
        print(f"パスが存在するか: {self.knowledge_base_path.exists()}")
        
        if not self.knowledge_base_path.exists():
            print(f"ナレッジベースパスが存在しません: {self.knowledge_base_path}")
            return []
        
        files = []
        for category_dir in self.knowledge_base_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                print(f"カテゴリディレクトリ: {category}")
                for file_path in category_dir.glob("*.md"):
                    print(f"ファイル発見: {file_path}")
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
        # 明確な質問のパターン（より柔軟に）
        industry_keywords = ["IT業界", "建設業界", "製造業界", "金融業界", "医療業界", "教育業界"]
        theme_keywords = ["動向", "最近", "ホット", "トピック", "課題", "事例", "研修", "育成", "改革", "防止", "パワハラ"]
        
        # 業界名 + 具体的なテーマがあれば明確
        has_industry = any(industry in user_input for industry in industry_keywords)
        has_theme = any(theme in user_input for theme in theme_keywords)
        
        if has_industry and has_theme:
            return False
        
        # 具体的なキーワードの組み合わせ
        specific_combinations = [
            ("IT業界", "動向"),
            ("IT業界", "最近"),
            ("IT業界", "ホット"),
            ("IT業界", "トピック"),
            ("建設業界", "パワハラ"),
            ("建設業界", "事例"),
            ("建設業界", "防止"),
            ("人材", "育成"),
            ("組織", "風土"),
            ("女性", "活躍"),
            ("働き方", "改革")
        ]
        
        for industry, theme in specific_combinations:
            if industry in user_input and theme in user_input:
                return False
        
        # 曖昧な表現のパターン
        ambiguous_words = ["知りたい", "教えて", "について", "どう", "何か", "どんな"]
        has_ambiguous = any(word in user_input for word in ambiguous_words)
        
        # 業界名 + 曖昧な表現の場合は曖昧
        if has_industry and has_ambiguous:
            return True
        
        # デフォルトは曖昧と判定
        return True
    
    def has_sufficient_information(self, user_input: str, answers: List[str]) -> bool:
        """十分な情報が得られたかを判定"""
        # ユーザーの反応をチェック（早期終了のサイン）
        negative_responses = ["いや", "そのまま", "いや、、", "いや、", "いや、そのまま", "そのままなんだけど"]
        for answer in answers:
            if any(negative in answer for negative in negative_responses):
                return True  # ユーザーが明確に答えたい場合は即答
        
        # 回答がない場合でも、十分に明確な質問の場合は即答
        if not answers:
            # 業界名 + 具体的なテーマがあれば十分
            industry_keywords = ["建設業界", "IT業界", "製造業界", "金融業界", "医療業界", "教育業界"]
            theme_keywords = ["動向", "最近", "ホット", "トピック", "課題", "事例", "研修", "育成", "改革", "防止", "パワハラ"]
            
            has_industry = any(industry in user_input for industry in industry_keywords)
            has_theme = any(theme in user_input for theme in theme_keywords)
            
            if has_industry and has_theme:
                return True
        
        # 回答の長さと内容をチェック
        total_length = sum(len(answer) for answer in answers)
        if total_length < 10:  # 合計10文字未満は不十分
            return False
        
        # 具体的なキーワードが含まれているかをチェック
        specific_keywords = [
            "建設業界", "パワハラ", "研修", "事例", "対策", "防止", "管理職", 
            "現場", "職人", "女性", "育成", "組織", "風土", "改革", "採用",
            "新卒", "働き方", "人材", "技能", "コミュニケーション", "安全",
            "IT業界", "動向", "最近", "ホット", "トピック", "トレンド"
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
        
        # 業界名が明確に含まれている場合
        industry_keywords = ["建設業界", "IT業界", "製造業界", "金融業界", "医療業界", "教育業界"]
        if any(industry in combined_text for industry in industry_keywords):
            # 業界名 + 具体的なテーマがあれば十分
            theme_keywords = ["動向", "課題", "事例", "研修", "育成", "改革", "防止"]
            if any(theme in combined_text for theme in theme_keywords):
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
        
        # ユーザーの反応をチェック
        negative_responses = ["いや", "そのまま", "いや、、", "いや、", "いや、そのまま", "そのままなんだけど"]
        for answer in answers:
            if any(negative in answer for negative in negative_responses):
                # ユーザーが明確に答えたい場合は早期完了
                return IntentQuestion(
                    question="理解しました。回答を生成します。",
                    question_number=question_number,
                    max_questions=3,
                    is_complete=True
                )
        
        prompt = f"""
あなたは人材開発・組織開発のコンサルタントです。ユーザーが知りたい具体的な情報を特定するための質問を生成してください。

ユーザーの最初の質問: {user_input}
これまでの回答: {', '.join(answers) if answers else 'なし'}

質問番号: {question_number}/3

以下の条件に従って質問を生成してください：
1. ユーザーが知りたい具体的な内容を掘り下げる質問をする
2. 質問は簡潔で自然な日本語にする
3. 質問のみを返し、余計な文章は含めないでください
4. 質問は1文で終わるようにしてください
5. ユーザーが「いや」「そのまま」などと答えた場合は、早期完了を示す

質問例：
「建設業界のどのような側面について知りたいですか？」
「パワハラ防止について、どのような具体的な情報をお探しですか？」
「IT業界の動向について、どのような分野に興味がありますか？」

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
        # ユーザーの反応をチェック
        negative_responses = ["いや", "そのまま", "いや、、", "いや、", "いや、そのまま", "そのままなんだけど"]
        for answer in answers:
            if any(negative in answer for negative in negative_responses):
                # ユーザーが明確に答えたい場合は早期完了
                return IntentQuestion(
                    question="理解しました。回答を生成します。",
                    question_number=question_number,
                    max_questions=3,
                    is_complete=True
                )
        
        context = f"{user_input}"
        if answers:
            context += f" {' '.join(answers)}"
        
        if question_number == 1:
            if "IT業界" in user_input or "IT" in user_input:
                question = "IT業界のどのような側面について知りたいですか？技術動向、人材育成、働き方改革など"
            elif "建設" in user_input:
                question = "建設業界のどのような側面について知りたいですか？市場動向、人材育成、現場環境など"
            elif "パワハラ" in user_input or "ハラスメント" in user_input:
                question = "パワハラ防止について、どのような具体的な情報をお探しですか？研修内容、事例、対策方法など"
            elif "人材" in user_input or "育成" in user_input:
                question = "人材育成について、どのような具体的な課題に興味がありますか？技能者不足、世代間ギャップ、研修制度など"
            else:
                question = f"「{user_input}」について、どのような情報をお探しですか？"
        elif question_number == 2:
            if "IT業界" in context or "IT" in context:
                question = "IT業界の動向について、どのような分野に興味がありますか？AI、DX、人材不足など"
            elif "パワハラ" in context or "ハラスメント" in context:
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
        text_lower = text.lower()
        
        # テーマキーワード（最優先）
        theme_keywords = {
            "パワハラ防止": ["パワハラ", "ハラスメント", "防止", "撲滅", "職場環境", "研修"],
            "女性活躍": ["女性", "活躍", "ダイバーシティ", "女性活躍"],
            "組織風土改革": ["組織", "風土", "文化", "組織風土", "組織文化", "改革"],
            "働き方改革": ["働き方", "改革", "ワークライフ", "バランス", "働き方改革"],
            "人材育成": ["人材", "育成", "教育", "研修", "スキル", "人材育成"],
            "管理職育成": ["管理職", "育成", "リーダー", "マネージャー", "管理職候補"],
            "若手離職防止": ["若手", "離職", "定着", "若手中堅", "若手離職"],
            "新人育成": ["新人", "育成", "新卒", "新人育成", "新入社員"],
            "コーチング": ["コーチング", "1on1", "面談", "コーチング1on1"],
            "人的資本開示": ["人的資本", "開示", "人的資本開示", "投資家"],
            "デジタル化推進": ["デジタル", "DX", "デジタル化", "デジタル変革"],
            "エンゲージメント向上": ["エンゲージメント", "従業員満足", "エンゲージメント向上"]
        }
        
        # 業界キーワード（中優先度）
        industry_keywords = {
            "建設業界": ["建設", "建築", "五洋建設", "大林組", "鹿島建設", "清水建設"],
            "IT業界": ["IT", "情報技術", "ソフトウェア", "システム", "プログラミング", "エンジニア"],
            "製造業界": ["製造", "工場", "生産", "製造業"],
            "金融業界": ["金融", "銀行", "三菱UFJ", "みずほ", "三井住友", "投資"],
            "小売業界": ["小売", "イオン", "セブン", "ファミマ", "ローソン", "コンビニ"],
            "運輸業界": ["運輸", "JR", "鉄道", "航空", "ANA", "JAL", "日本郵船"],
            "医療業界": ["医療", "病院", "製薬", "武田薬品", "アステラス", "看護"],
            "教育業界": ["教育", "大学", "早稲田", "東京大学", "学校", "教員"],
            "エネルギー業界": ["エネルギー", "電力", "東京電力", "関西電力", "ガス", "石油"]
        }
        
        # 1. テーマキーワードの抽出（最優先）
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                keywords.append(theme)
        
        # 2. 業界キーワードの抽出（中優先度）
        for industry, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                keywords.append(industry)
                break  # 1つの業界のみを抽出
        
        # 3. 具体的なキーワードの抽出
        specific_keywords = [
            "現場", "職人", "技能者", "技能伝承", "バリューチェーン",
            "事例", "実践", "研修", "対策", "防止", "撲滅",
            "動向", "最近", "ホット", "トピック", "トレンド",
            "課題", "問題", "解決", "改善", "向上"
        ]
        
        for keyword in specific_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
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
        print(f"検索クエリ: {query}")
        
        # クエリを単語に分割（日本語と英語の両方に対応）
        query_words = re.findall(r'[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', query.lower())
        
        # 業界キーワードの定義（大幅に拡充）
        industry_keywords = {
            "自動車": ["自動車", "車", "トヨタ", "ホンダ", "日産", "マツダ", "スバル", "三菱", "次世代リーダー"],
            "食品": ["食品", "食", "明治", "味の素", "キッコーマン", "カルビー", "江崎グリコ"],
            "化学": ["化学", "三菱化学", "住友化学", "旭化成", "東レ", "帝人"],
            "建設": ["建設", "建築", "五洋建設", "大林組", "鹿島建設", "清水建設"],
            "IT": ["IT", "情報技術", "ソフトウェア", "システム", "プログラミング", "エンジニア"],
            "製造": ["製造", "工場", "生産", "製造業"],
            "金融": ["金融", "銀行", "三菱UFJ", "みずほ", "三井住友", "投資"],
            "小売": ["小売", "イオン", "セブン", "ファミマ", "ローソン", "コンビニ"],
            "運輸": ["運輸", "JR", "鉄道", "航空", "ANA", "JAL", "日本郵船"],
            "医療": ["医療", "病院", "製薬", "武田薬品", "アステラス", "看護"],
            "教育": ["教育", "大学", "早稲田", "東京大学", "学校", "教員"],
            "エネルギー": ["エネルギー", "電力", "東京電力", "関西電力", "中部電力"]
        }
        
        # テーマキーワードの定義（最優先）
        theme_keywords = {
            "パワハラ防止": ["パワハラ", "ハラスメント", "防止", "撲滅", "職場環境", "研修"],
            "女性活躍": ["女性", "活躍", "ダイバーシティ", "女性活躍"],
            "組織風土改革": ["組織", "風土", "文化", "組織風土", "組織文化", "改革"],
            "働き方改革": ["働き方", "改革", "ワークライフ", "バランス", "働き方改革"],
            "人材育成": ["人材", "育成", "教育", "研修", "スキル", "人材育成"],
            "管理職育成": ["管理職", "育成", "リーダー", "マネージャー", "管理職候補"],
            "若手離職防止": ["若手", "離職", "定着", "若手中堅", "若手離職"],
            "新人育成": ["新人", "育成", "新卒", "新人育成", "新入社員"],
            "コーチング": ["コーチング", "1on1", "面談", "コーチング1on1"],
            "エンゲージメント": ["エンゲージメント", "従業員満足", "モチベーション"],
            "デジタル化": ["デジタル", "DX", "デジタル化", "デジタル変革"],
            "AI人材": ["AI", "人工知能", "AI人材", "デジタル人材"],
            "人的資本": ["人的資本", "人材投資", "人的資本開示"],
            "中堅育成": ["中堅", "中堅層", "中堅の意識改革"],
            "経営候補": ["経営", "経営候補", "経営層", "次世代リーダー"],
            "部長層": ["部長", "部長層", "管理職", "リーダーシップ"]
        }
        
        for file_info in files:
            score = 0
            file_path = file_info['path']
            file_title = file_info['title']
            file_content = file_info.get('content', '')
            
            # テーママッチング（最優先：300点）
            matched_theme = None
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in query.lower() or keyword in file_title.lower() or keyword in file_content.lower():
                        score += 300
                        matched_theme = theme
                        break
                if matched_theme:
                    break
            
            # 業界マッチング（中優先度：150点）
            matched_industry = None
            for industry, keywords in industry_keywords.items():
                for keyword in keywords:
                    if keyword in query.lower() or keyword in file_title.lower() or keyword in file_content.lower():
                        score += 150
                        matched_industry = industry
                        break
                if matched_industry:
                    break
            
            # テーマ+業界の組み合わせボーナス（100点）
            if matched_theme and matched_industry:
                score += 100
            
            # テーマミスマッチのペナルティ（-200点）
            if matched_theme:
                # クエリにテーマキーワードがあるが、ファイルにない場合
                query_has_theme = any(theme in query.lower() for theme_list in theme_keywords.values() for theme in theme_list)
                file_has_theme = any(theme in file_title.lower() or theme in file_content.lower() for theme_list in theme_keywords.values() for theme in theme_list)
                if query_has_theme and not file_has_theme:
                    score -= 200
            
            # 完全一致キーワードマッチング（10点）
            for word in query_words:
                if word in file_title.lower() or word in file_content.lower():
                    score += 10
            
            # カテゴリマッチング（5点）
            if '能力開発テーマ' in file_path and any(theme in query.lower() for theme_list in theme_keywords.values() for theme in theme_list):
                score += 5
            elif '業界別' in file_path and any(industry in query.lower() for industry_list in industry_keywords.values() for industry in industry_list):
                score += 5
            
            # ファイルサイズボーナス（小さいファイルを優先：-1点/1000文字）
            file_size_penalty = len(file_content) // 1000
            score -= file_size_penalty
            
            if score > 0:
                results.append({
                    'path': file_path,
                    'title': file_title,
                    'category': file_info.get('category', ''),
                    'keywords': file_info.get('keywords', []),
                    'description': file_info.get('description', ''),
                    'file_size': file_info.get('file_size', 0),
                    'last_modified': file_info.get('last_modified', ''),
                    'score': score
                })
        
        # スコアで降順ソート
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"検索結果数: {len(results)}")
        for result in results[:5]:  # 上位5件を表示
            print(f"  {result['title']}: {result['score']}点")
        
        return results

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
    
    # 業界とテーマを抽出
    query_industry = None
    query_theme = None
    
    # 業界キーワードの定義
    industry_keywords = {
        "建設": ["建設", "建築", "五洋建設", "大林組", "鹿島建設", "清水建設"],
        "IT": ["IT", "情報技術", "ソフトウェア", "システム", "プログラミング", "エンジニア"],
        "製造": ["製造", "工場", "生産", "製造業"],
        "金融": ["金融", "銀行", "三菱UFJ", "みずほ", "三井住友", "投資"],
        "小売": ["小売", "イオン", "セブン", "ファミマ", "ローソン", "コンビニ"],
        "運輸": ["運輸", "JR", "鉄道", "航空", "ANA", "JAL", "日本郵船"],
        "医療": ["医療", "病院", "製薬", "武田薬品", "アステラス", "看護"],
        "教育": ["教育", "大学", "早稲田", "東京大学", "学校", "教員"],
        "エネルギー": ["エネルギー", "電力", "東京電力", "関西電力", "中部電力"]
    }
    
    # テーマキーワードの定義
    theme_keywords = {
        "パワハラ防止": ["パワハラ", "ハラスメント", "防止", "撲滅", "職場環境", "研修"],
        "女性活躍": ["女性", "活躍", "ダイバーシティ", "女性活躍"],
        "組織風土改革": ["組織", "風土", "文化", "組織風土", "組織文化", "改革"],
        "働き方改革": ["働き方", "改革", "ワークライフ", "バランス", "働き方改革"],
        "人材育成": ["人材", "育成", "教育", "研修", "スキル", "人材育成"],
        "管理職育成": ["管理職", "育成", "リーダー", "マネージャー", "管理職候補"],
        "若手離職防止": ["若手", "離職", "定着", "若手中堅", "若手離職"],
        "新人育成": ["新人", "育成", "新卒", "新人育成", "新入社員"],
        "コーチング": ["コーチング", "1on1", "面談", "コーチング1on1"],
        "エンゲージメント": ["エンゲージメント", "従業員満足", "モチベーション"],
        "デジタル化": ["デジタル", "DX", "デジタル化", "デジタル変革"],
        "AI人材": ["AI", "人工知能", "AI人材", "デジタル人材"],
        "人的資本": ["人的資本", "人材投資", "人的資本開示"],
        "中堅育成": ["中堅", "中堅層", "中堅の意識改革"],
        "経営候補": ["経営", "経営候補", "経営層", "次世代リーダー"],
        "部長層": ["部長", "部長層", "管理職", "リーダーシップ"]
    }
    
    # クエリから業界とテーマを特定
    for industry, keywords in industry_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            query_industry = industry
            break
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            query_theme = theme
            break
    
    # ファイルを業界別に分類
    same_industry_files = []
    different_industry_files = []
    
    for file in knowledge_files[:5]:  # 最大5件まで
        file_industry = None
        for industry, keywords in industry_keywords.items():
            if any(keyword in file['title'] for keyword in keywords):
                file_industry = industry
                break
        
        if file_industry == query_industry:
            same_industry_files.append(file)
        else:
            different_industry_files.append(file)
    
    # 同じ業界のファイルを優先的に表示
    for file in same_industry_files:
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
    
    # 異なる業界のファイルを参考として表示
    if different_industry_files and query_theme:
        response_parts.append(f"\n【参考情報】{query_theme}について、他の業界の事例も参考になります：\n")
        
        for file in different_industry_files[:2]:  # 最大2件まで
            try:
                rel_path = file['path'].strip()
                if rel_path.startswith("KNOWLEDGE/"):
                    file_path = Path("..") / rel_path
                else:
                    file_path = Path("../KNOWLEDGE") / rel_path
                
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    summary = []
                    
                    # テーマに関連する情報を抽出
                    for i, line in enumerate(lines[:15]):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # テーマに関連するキーワードを含む行を抽出
                        theme_keywords_list = theme_keywords.get(query_theme, [])
                        if any(keyword in line for keyword in theme_keywords_list):
                            summary.append(f"• {line}")
                            if len(summary) >= 3:  # 最大3行まで
                                break
                    
                    response_parts.append(f"【{file['title']}】（参考）")
                    if summary:
                        response_parts.extend(summary)
                    else:
                        description = file.get('description', '参考情報が含まれています')
                        if len(description) > 80:
                            description = description[:80] + "..."
                        response_parts.append(f"• {description}")
                    response_parts.append("")
            except Exception as e:
                print(f"フォールバック: 参考ファイル読み込みエラー {file['path']}: {e}")
    
    result = "\n".join(response_parts)
    print(f"フォールバック: 生成された応答の長さ: {len(result)}")
    return result

@app.post("/intent/start", response_model=IntentClarificationResponse)
async def start_intent_clarification(request: IntentClarificationRequest):
    """意図特定を開始"""
    try:
        print(f"意図特定開始: {request.user_input}")
        
        # 十分な情報が既にあるかチェック
        has_sufficient = intent_clarification_service.has_sufficient_information(request.user_input, [])
        print(f"十分な情報があるか: {has_sufficient}")
        
        if has_sufficient:
            # 十分な情報がある場合は即答
            question = IntentQuestion(
                question="十分な情報が得られました。回答を生成します。",
                question_number=1,
                max_questions=3,
                is_complete=True
            )
            print("即答モードで回答を生成")
            return IntentClarificationResponse(
                question=question,
                is_ambiguous=False
            )
        
        is_ambiguous = intent_clarification_service.is_ambiguous(request.user_input)
        print(f"曖昧かどうか: {is_ambiguous}")
        
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