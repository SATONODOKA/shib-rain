#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ナレッジ階層化自動化スクリプト
INBOXのアポメモを自動的にタグ付けし、業界別・能力開発テーマ別にナレッジ化するシステム
"""

import os
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class KnowledgeHierarchySystem:
    """ナレッジ階層化システムのメインクラス"""
    
    def __init__(self, vault_path: str = "."):
        self.vault_path = Path(vault_path)
        self.inbox_path = self.vault_path / "INBOX"
        self.knowledge_path = self.vault_path / "KNOWLEDGE"
        self.processing_path = self.vault_path / "PROCESSING"
        self.completed_path = self.processing_path / "完了"
        
        # タグ設定の読み込み
        self.tag_config = self.load_tag_config()
        
        # ディレクトリの初期化
        self.initialize_directories()
        
        # 処理済みファイルの追跡
        self.processed_files = self.load_processed_files()
    
    def load_tag_config(self) -> Dict:
        """タグ設定ファイルを読み込み"""
        config_path = self.vault_path / "tag_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # デフォルト設定
            return self.get_default_tag_config()
    
    def get_default_tag_config(self) -> Dict:
        """デフォルトのタグ設定を返す"""
        return {
            "業界": {
                "建設業界": ["建設", "建築", "土木", "工事", "現場", "五洋建設", "鹿島建設", "清水建設"],
                "製造業界": ["製造", "工場", "生産", "トヨタ", "日立", "三菱", "クボタ"],
                "IT業界": ["IT", "システム", "ソフトウェア", "プログラミング", "NTT", "ソフトバンク"],
                "金融業界": ["銀行", "金融", "保険", "三菱UFJ", "みずほ", "三井住友"],
                "小売業界": ["小売", "店舗", "イオン", "ユニクロ", "セブン"],
                "医療業界": ["医療", "病院", "看護", "東京大学病院", "武田薬品"],
                "教育業界": ["教育", "学校", "大学", "早稲田", "東京大学"],
                "運輸業界": ["運輸", "物流", "JR", "ANA", "日本郵船"],
                "エネルギー業界": ["エネルギー", "電力", "東京電力", "ガス"],
                "通信業界": ["通信", "携帯", "5G", "NTT", "KDDI"],
                "自動車業界": ["自動車", "車", "トヨタ", "ホンダ", "日産"],
                "化学業界": ["化学", "三菱化学", "旭化成"],
                "食品業界": ["食品", "明治", "味の素", "キッコーマン"]
            },
            "能力開発テーマ": {
                "組織風土改革": ["組織風土", "風土", "文化", "改革", "変革"],
                "女性活躍": ["女性", "活躍", "ジェンダー", "ダイバーシティ"],
                "働き方改革": ["働き方", "ワークライフ", "フレックス", "リモート"],
                "採用戦略": ["採用", "人材確保", "新卒", "中途"],
                "人材育成": ["育成", "研修", "教育", "スキル"],
                "評価制度": ["評価", "人事", "制度", "査定"],
                "コミュニケーション": ["コミュニケーション", "連携", "情報共有"],
                "意思決定": ["意思決定", "決断", "判断", "プロセス"],
                "プロジェクト管理": ["プロジェクト", "管理", "進捗", "スケジュール"],
                "リーダーシップ": ["リーダー", "マネジメント", "統率"],
                "コーチング": ["コーチング", "1on1", "面談"],
                "若手の自立": ["若手", "新人", "自立", "成長"],
                "エンゲージメント": ["エンゲージメント", "満足度", "モチベーション"],
                "デジタル化推進": ["デジタル", "DX", "IT化", "システム化"],
                "デジタル変革": ["デジタル変革", "DX", "変革", "革新"],
                "AI人材採用": ["AI", "人工知能", "人材", "採用"],
                "中堅の意識改革": ["中堅", "意識", "改革", "変革"],
                "人的資本開示": ["人的資本", "開示", "ESG", "サステナビリティ"],
                "次世代リーダー育成": ["次世代", "リーダー", "育成", "後継者"],
                "管理職候補育成": ["管理職", "候補", "育成", "マネージャー"],
                "新任管理職育成": ["新任", "管理職", "育成", "新規"],
                "経営候補選抜育成": ["経営", "候補", "選抜", "育成"],
                "若手中堅離職防止": ["離職", "防止", "定着", "若手", "中堅"],
                "部長層の育成": ["部長", "育成", "マネジメント"],
                "ハラスメント": ["ハラスメント", "パワハラ", "セクハラ", "防止"]
            },
            "バリューチェーン": {
                "現場職人": ["現場", "職人", "作業員", "技術者"],
                "現場監督": ["監督", "現場管理", "工事監督"],
                "設計者": ["設計", "エンジニア", "技術者"],
                "デベロッパー": ["開発", "プログラマー", "エンジニア"],
                "コンサルタント": ["コンサル", "アドバイザー"],
                "管理職": ["管理職", "マネージャー", "課長", "部長"],
                "経営層": ["経営", "役員", "社長", "取締役"]
            },
            "問題・課題": {
                "人材不足": ["人材不足", "人手不足", "採用難"],
                "長時間労働": ["長時間", "残業", "過労"],
                "離職率": ["離職", "定着", "退職"],
                "コミュニケーション不足": ["コミュニケーション", "連携不足"],
                "意思決定遅延": ["意思決定", "遅延", "判断"],
                "現場と本社の乖離": ["現場", "本社", "乖離", "ギャップ"],
                "世代間ギャップ": ["世代", "ギャップ", "価値観"],
                "女性参入障壁": ["女性", "参入", "障壁", "ハラスメント"]
            },
            "施策・打ち手": {
                "フレックスタイム": ["フレックス", "時差出勤"],
                "リモートワーク": ["リモート", "テレワーク", "在宅"],
                "メンター制度": ["メンター", "指導", "育成"],
                "評価制度改革": ["評価", "制度", "改革"],
                "研修プログラム": ["研修", "プログラム", "教育"],
                "組織開発": ["組織", "開発", "OD"],
                "ICT活用": ["ICT", "IT", "システム", "活用"],
                "プロセス改善": ["プロセス", "改善", "効率化"],
                "組織再編": ["組織", "再編", "構造改革"],
                "環境整備": ["環境", "整備", "改善"]
            },
            "結果": {
                "成功事例": ["成功", "効果", "改善"],
                "失敗事例": ["失敗", "課題", "問題"],
                "効果測定済み": ["効果", "測定", "検証"],
                "継続中": ["継続", "進行中", "実施中"],
                "中止": ["中止", "停止", "終了"],
                "他社展開済み": ["他社", "展開", "横展開"]
            }
        }
    
    def initialize_directories(self):
        """必要なディレクトリを作成"""
        directories = [
            self.inbox_path,
            self.knowledge_path / "業界別",
            self.knowledge_path / "能力開発テーマ",
            self.processing_path,
            self.completed_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"ディレクトリ確認/作成: {directory}")
    
    def load_processed_files(self) -> Set[str]:
        """処理済みファイルのリストを読み込み"""
        processed_file = self.vault_path / "processed_files.json"
        if processed_file.exists():
            with open(processed_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_files(self):
        """処理済みファイルのリストを保存"""
        processed_file = self.vault_path / "processed_files.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)
    
    def extract_tags_from_content(self, content: str) -> Dict[str, List[str]]:
        """コンテンツからタグを抽出"""
        tags = {category: [] for category in self.tag_config.keys()}
        
        for category, tag_rules in self.tag_config.items():
            for tag_name, keywords in tag_rules.items():
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        if tag_name not in tags[category]:
                            tags[category].append(tag_name)
                        break
        
        return tags
    
    def extract_company_name(self, filename: str) -> str:
        """ファイル名から会社名を抽出"""
        # ファイル名の形式: 会社名_アポメモ_日付_テーマ.md
        parts = filename.replace('.md', '').split('_')
        if len(parts) >= 1:
            return parts[0]
        return "不明"
    
    def extract_date(self, filename: str) -> str:
        """ファイル名から日付を抽出"""
        # 日付のパターンを検索
        date_pattern = r'(\d{8})'
        match = re.search(date_pattern, filename)
        if match:
            date_str = match.group(1)
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return "不明"
    
    def extract_theme(self, filename: str) -> str:
        """ファイル名からテーマを抽出"""
        parts = filename.replace('.md', '').split('_')
        if len(parts) >= 4:
            return parts[3]
        return "不明"
    
    def create_knowledge_content(self, file_path: Path, tags: Dict[str, List[str]], 
                                company: str, date: str, theme: str) -> str:
        """ナレッジファイルのコンテンツを生成"""
        
        # 元ファイルの内容を読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 業界タグを取得
        industries = tags.get("業界", [])
        industry = industries[0] if industries else "不明"
        
        # 能力開発テーマを取得
        themes = tags.get("能力開発テーマ", [])
        primary_theme = themes[0] if themes else theme
        
        # 問題・課題を取得
        problems = tags.get("問題・課題", [])
        
        # 施策・打ち手を取得
        solutions = tags.get("施策・打ち手", [])
        
        # 結果を取得
        results = tags.get("結果", [])
        
        # バリューチェーンを取得
        value_chains = tags.get("バリューチェーン", [])
        
        content = f"""# {industry} - {primary_theme}

## 📋 基本情報

- **会社名**: {company}
- **日付**: {date}
- **テーマ**: {primary_theme}
- **元ファイル**: [[{file_path.name}]]

## 🏷️ タグ情報

### 業界
{', '.join(industries) if industries else '不明'}

### 能力開発テーマ
{', '.join(themes) if themes else '不明'}

### バリューチェーン
{', '.join(value_chains) if value_chains else '不明'}

### 問題・課題
{', '.join(problems) if problems else '不明'}

### 施策・打ち手
{', '.join(solutions) if solutions else '不明'}

### 結果
{', '.join(results) if results else '不明'}

## 📝 元コンテンツ

```
{original_content}
```

## 🔗 関連ナレッジ

<!-- 関連するナレッジファイルへのリンクが自動生成されます -->

---
*最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*自動生成: ナレッジ階層化システム*
"""
        
        return content
    
    def create_industry_knowledge(self, industry: str, files: List[Path]) -> str:
        """業界別ナレッジファイルを生成"""
        
        content = f"""# {industry} - 業界一般

## 📊 業界概要

{industry}における人材開発・組織開発の課題と取り組みについてまとめています。

## 📁 関連ファイル

"""
        
        for file_path in files:
            company = self.extract_company_name(file_path.name)
            date = self.extract_date(file_path.name)
            theme = self.extract_theme(file_path.name)
            
            content += f"- [[{file_path.name}]] - {company} ({date}) - {theme}\n"
        
        content += f"""

## 🏷️ 主要タグ

### 能力開発テーマ
<!-- この業界でよく見られるテーマが自動生成されます -->

### 問題・課題
<!-- この業界特有の課題が自動生成されます -->

### 施策・打ち手
<!-- この業界で効果的な施策が自動生成されます -->

---
*最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*自動生成: ナレッジ階層化システム*
"""
        
        return content
    
    def create_theme_knowledge(self, theme: str, industry: str, files: List[Path]) -> str:
        """テーマ別ナレッジファイルを生成"""
        
        content = f"""# {theme} - {industry}

## 🎯 テーマ概要

{industry}における{theme}の取り組みについてまとめています。

## 📁 関連ファイル

"""
        
        for file_path in files:
            company = self.extract_company_name(file_path.name)
            date = self.extract_date(file_path.name)
            
            content += f"- [[{file_path.name}]] - {company} ({date})\n"
        
        content += f"""

## 📊 業界別分析

### {industry}の特徴
<!-- この業界における{theme}の特徴が自動生成されます -->

### 成功要因
<!-- 成功事例の共通要因が自動生成されます -->

### 課題・注意点
<!-- 失敗事例や課題が自動生成されます -->

## 🔗 関連テーマ

<!-- 関連するテーマへのリンクが自動生成されます -->

---
*最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*自動生成: ナレッジ階層化システム*
"""
        
        return content
    
    def process_inbox_files(self):
        """INBOXフォルダのファイルを処理"""
        if not self.inbox_path.exists():
            logging.warning("INBOXフォルダが存在しません")
            return
        
        # 新規ファイルを検出
        new_files = []
        for file_path in self.inbox_path.glob("*.md"):
            if file_path.name not in self.processed_files:
                new_files.append(file_path)
        
        if not new_files:
            logging.info("処理対象の新規ファイルがありません")
            return
        
        logging.info(f"処理対象ファイル数: {len(new_files)}")
        
        # 各ファイルを処理
        for file_path in new_files:
            try:
                self.process_single_file(file_path)
                self.processed_files.add(file_path.name)
            except Exception as e:
                logging.error(f"ファイル処理エラー {file_path.name}: {e}")
        
        # 処理済みファイルリストを保存
        self.save_processed_files()
        
        # ナレッジファイルを生成
        self.generate_knowledge_files()
    
    def process_single_file(self, file_path: Path):
        """単一ファイルを処理"""
        logging.info(f"ファイル処理中: {file_path.name}")
        
        # ファイル内容を読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # タグを抽出
        tags = self.extract_tags_from_content(content)
        
        # ファイル情報を抽出
        company = self.extract_company_name(file_path.name)
        date = self.extract_date(file_path.name)
        theme = self.extract_theme(file_path.name)
        
        # タグ付きコンテンツを生成
        tagged_content = self.create_tagged_content(content, tags, company, date, theme)
        
        # タグ付きファイルを保存
        tagged_file_path = self.processing_path / f"tagged_{file_path.name}"
        with open(tagged_file_path, 'w', encoding='utf-8') as f:
            f.write(tagged_content)
        
        # 完了フォルダに移動
        completed_file_path = self.completed_path / file_path.name
        shutil.move(str(file_path), str(completed_file_path))
        
        logging.info(f"ファイル処理完了: {file_path.name}")
    
    def create_tagged_content(self, content: str, tags: Dict[str, List[str]], 
                             company: str, date: str, theme: str) -> str:
        """タグ付きコンテンツを生成"""
        
        # タグ情報を文字列化
        tag_info = []
        for category, tag_list in tags.items():
            if tag_list:
                tag_info.append(f"## {category}\n{', '.join(tag_list)}\n")
        
        tagged_content = f"""# {company} - {theme}

## 📋 基本情報

- **会社名**: {company}
- **日付**: {date}
- **テーマ**: {theme}

## 🏷️ 抽出タグ

{chr(10).join(tag_info)}

## 📝 元コンテンツ

{content}

---
*自動タグ付け: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return tagged_content
    
    def generate_knowledge_files(self):
        """ナレッジファイルを生成"""
        logging.info("ナレッジファイル生成開始")
        
        # 完了フォルダのファイルを分析
        completed_files = list(self.completed_path.glob("*.md"))
        
        if not completed_files:
            logging.info("完了フォルダにファイルがありません")
            return
        
        # 業界別にファイルをグループ化
        industry_files = {}
        theme_industry_files = {}
        
        for file_path in completed_files:
            # ファイル内容を読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # タグを抽出
            tags = self.extract_tags_from_content(content)
            
            # 業界を取得
            industries = tags.get("業界", [])
            if industries:
                industry = industries[0]
                if industry not in industry_files:
                    industry_files[industry] = []
                industry_files[industry].append(file_path)
            
            # テーマ×業界を取得
            themes = tags.get("能力開発テーマ", [])
            if themes and industries:
                theme = themes[0]
                industry = industries[0]
                key = f"{theme}_{industry}"
                if key not in theme_industry_files:
                    theme_industry_files[key] = []
                theme_industry_files[key].append(file_path)
        
        # 業界別ナレッジファイルを生成
        for industry, files in industry_files.items():
            content = self.create_industry_knowledge(industry, files)
            file_path = self.knowledge_path / "業界別" / f"{industry}_業界一般.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"業界別ナレッジ生成: {file_path}")
        
        # テーマ×業界別ナレッジファイルを生成
        for key, files in theme_industry_files.items():
            theme, industry = key.split('_', 1)
            content = self.create_theme_knowledge(theme, industry, files)
            file_path = self.knowledge_path / "能力開発テーマ" / f"{theme}_{industry}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"テーマ別ナレッジ生成: {file_path}")
        
        logging.info("ナレッジファイル生成完了")
    
    def add_tag_rule(self, category: str, tag_name: str, keywords: List[str]):
        """タグルールを追加"""
        if category not in self.tag_config:
            self.tag_config[category] = {}
        
        self.tag_config[category][tag_name] = keywords
        
        # 設定ファイルを保存
        config_path = self.vault_path / "tag_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.tag_config, f, ensure_ascii=False, indent=2)
        
        logging.info(f"タグルール追加: {category} - {tag_name}")

def main():
    """メイン関数"""
    print("ナレッジ階層化システムを開始します...")
    
    # システムを初期化
    system = KnowledgeHierarchySystem()
    
    # INBOXファイルを処理
    system.process_inbox_files()
    
    print("ナレッジ階層化システムが完了しました。")

if __name__ == "__main__":
    main() 