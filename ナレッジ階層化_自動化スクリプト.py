#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ナレッジ階層化システム
INBOXのアポメモを自動的にタグ付けし、業界別・能力開発テーマ別にナレッジ化するシステム

プロセス:
1. INBOX監視: 新規ファイルの自動検出
2. 自動タグ付け: 業界・テーマ・課題・施策の自動抽出
3. ナレッジ生成: 業界別・能力開発テーマ別のナレッジファイル作成
4. 関連性発見: 類似事例・他社事例との関連付け

特徴:
- タグは動的に追加・更新可能（拡張性重視）
- 新しい業界・テーマ・課題が追加されても自動対応
- タグの重み付け・関連性も管理可能
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set

class KnowledgeHierarchySystem:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.inbox_path = self.vault_path / "INBOX"
        self.knowledge_path = self.vault_path / "KNOWLEDGE"
        
        # 業界別フォルダ
        self.industry_path = self.knowledge_path / "業界別"
        # 能力開発テーマ別フォルダ
        self.theme_path = self.knowledge_path / "能力開発テーマ"
        
        # タグ設定ファイルのパス
        self.tag_config_path = self.vault_path / "tag_config.json"
        
        # タグ付けルール（初期設定 - 今後拡張可能）
        self.tag_rules = self.load_tag_rules()

    def load_tag_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """タグ設定をファイルから読み込み（存在しない場合は初期設定を使用）"""
        if self.tag_config_path.exists():
            try:
                with open(self.tag_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 初期設定（今後どんどん拡張されることを前提）
        initial_rules = {
            '業界': {
                '建設業界': ['五洋建設', '建設現場', '工事現場', '建築現場', '土木工事'],
                '製造業': ['製造工場', '生産工場', 'メーカー工場'],
                'IT業界': ['IT企業', 'システム開発', 'ソフトウェア開発'],
                '金融業界': ['銀行', '保険会社', '証券会社'],
                'サービス業': ['サービス企業', 'ホテル', '飲食店'],
                '小売業': ['小売店', '流通企業', '販売店'],
                '医療業界': ['病院', 'クリニック', '医療機関'],
                '教育業界': ['学校', '大学', '教育機関'],
                '運輸業界': ['運輸会社', '物流企業', '配送会社']
            },
            '能力開発テーマ': {
                '組織風土改革': ['組織風土', '風土改革', '組織文化改革'],
                '女性活躍': ['女性活躍', '女性管理職', 'ダイバーシティ'],
                '働き方改革': ['働き方改革', 'フレックスタイム', 'テレワーク'],
                '採用戦略': ['採用戦略', '新卒採用', '人材確保'],
                '人材育成': ['人材育成', 'メンター制度', 'OJT'],
                '評価制度': ['評価制度', '人事評価', '360度評価'],
                'コミュニケーション': ['コミュニケーション', '情報共有'],
                '意思決定': ['意思決定', '決裁', '稟議'],
                'プロジェクト管理': ['プロジェクト管理', 'PM', '進捗管理'],
                'リーダーシップ': ['リーダーシップ', '管理職育成'],
                'コーチング': ['コーチング', 'メンタリング'],
                '若手の自立': ['若手', '新人', '自立'],
                'エンゲージメント': ['エンゲージメント', '従業員満足'],
                'コンサル': ['コンサル', '外部支援']
            },
            'バリューチェーン': {
                '現場職人': ['現場職人', '技能者', '大工', '左官'],
                '現場監督': ['現場監督', '安全管理'],
                '設計者': ['設計者', '建築設計', '構造設計'],
                'デベロッパー': ['デベロッパー', '開発'],
                'コンサルタント': ['コンサルタント', '監理'],
                '管理職': ['管理職', '部長', '課長'],
                '経営層': ['経営層', '役員', '取締役']
            },
            '問題・課題': {
                '人材不足': ['人材不足', '人手不足', '技能者不足'],
                '長時間労働': ['長時間労働', '残業', '過労'],
                '離職率': ['離職率', '退職', '定着率'],
                'コミュニケーション不足': ['コミュニケーション不足', '情報共有不足'],
                '意思決定遅延': ['意思決定遅延', '決裁遅延'],
                '現場と本社の乖離': ['現場と本社', '現場と本部'],
                '世代間ギャップ': ['世代間', 'ギャップ'],
                '女性参入障壁': ['女性参入', '女性障壁']
            },
            '施策・打ち手': {
                'フレックスタイム': ['フレックスタイム', '時差出勤'],
                'リモートワーク': ['リモートワーク', '在宅勤務'],
                'メンター制度': ['メンター制度', 'メンタリング'],
                '評価制度改革': ['評価制度改革', '人事制度改革'],
                '研修プログラム': ['研修プログラム', '教育プログラム'],
                '組織開発': ['組織開発', '組織改革'],
                'ICT活用': ['ICT活用', 'IT活用'],
                'プロセス改善': ['プロセス改善', '業務改善'],
                '組織再編': ['組織再編', '組織変更'],
                '環境整備': ['環境整備', '設備整備']
            },
            '結果': {
                '成功事例': ['成功', '効果', '成果'],
                '失敗事例': ['失敗', '問題', '課題'],
                '効果測定済み': ['効果測定', 'KPI'],
                '継続中': ['継続中', '進行中'],
                '中止': ['中止', '停止'],
                '他社展開済み': ['他社展開', '横展開']
            }
        }
        
        # 初期設定を保存
        self.save_tag_rules(initial_rules)
        return initial_rules

    def save_tag_rules(self, rules: Dict[str, Dict[str, List[str]]]):
        """タグ設定をファイルに保存"""
        with open(self.tag_config_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)

    def add_tag_rule(self, category: str, tag: str, keywords: List[str]):
        """新しいタグルールを追加（拡張性のための機能）"""
        if category not in self.tag_rules:
            self.tag_rules[category] = {}
        
        self.tag_rules[category][tag] = keywords
        self.save_tag_rules(self.tag_rules)
        print(f"新しいタグを追加しました: {category}/{tag}")

    def get_all_tags(self) -> Set[str]:
        """全タグを取得（重複除去）"""
        all_tags = set()
        for category_rules in self.tag_rules.values():
            for tag in category_rules.keys():
                all_tags.add(tag)
        return all_tags

    def get_tag_statistics(self) -> Dict[str, int]:
        """タグの統計情報を取得"""
        stats = {}
        for category, rules in self.tag_rules.items():
            stats[category] = len(rules)
        return stats

    def analyze_file(self, file_path: Path) -> Dict[str, List[str]]:
        """ファイルを分析してタグを抽出（拡張可能なタグシステム）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tags = {}
        
        # 各カテゴリのタグを抽出（新しいカテゴリが追加されても自動対応）
        for category, rules in self.tag_rules.items():
            category_tags = []
            for tag, keywords in rules.items():
                for keyword in keywords:
                    if keyword in content:
                        category_tags.append(tag)
                        break
            tags[category] = category_tags
        
        return tags

    def determine_knowledge_files(self, file_path: Path, tags: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """ナレッジファイルの種類と配置場所を決定（新しい業界・テーマにも自動対応）"""
        knowledge_files = []
        
        # 業界別ナレッジの決定
        industries = tags.get('業界', [])
        value_chains = tags.get('バリューチェーン', [])
        
        for industry in industries:
            # 業界一般のナレッジ（既存ファイルがある場合のみ）
            industry_general_path = self.industry_path / f"{industry}_業界一般.md"
            if industry_general_path.exists():
                knowledge_files.append(('業界別', f"{industry}_業界一般"))
            
            # バリューチェーン別のナレッジ（既存ファイルがある場合のみ）
            for value_chain in value_chains:
                value_chain_path = self.industry_path / f"{industry}_{value_chain}.md"
                if value_chain_path.exists():
                    knowledge_files.append(('業界別', f"{industry}_{value_chain}"))
        
        # 能力開発テーマ別ナレッジの決定
        themes = tags.get('能力開発テーマ', [])
        
        for theme in themes:
            for industry in industries:
                theme_path = self.theme_path / f"{theme}_{industry}.md"
                if theme_path.exists():
                    knowledge_files.append(('能力開発テーマ', f"{theme}_{industry}"))
        
        return knowledge_files

    def create_knowledge_content(self, folder_type: str, filename: str, source_file: Path, tags: Dict[str, List[str]]) -> str:
        """ナレッジファイルの内容を作成"""
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本情報を抽出
        date_match = re.search(r'\*\*日時\*\*: (.+)', content)
        date = date_match.group(1) if date_match else '不明'
        
        # 担当者情報を抽出
        participants = ''
        participants_match = re.search(r'\*\*参加者\*\*:([\s\S]+?)(\n\*\*|\n#|\n---|$)', content)
        if participants_match:
            participants = participants_match.group(1).strip().replace('\n', ' ')
        
        # ファイル名から情報を解析
        if folder_type == '業界別':
            if '_業界一般' in filename:
                industry = filename.replace('_業界一般', '')
                title = f"{industry}_業界一般"
                description = f"{industry}業界の全体像、特徴、課題、動向をまとめたナレッジ"
            else:
                parts = filename.split('_', 1)
                industry = parts[0]
                value_chain = parts[1]
                title = f"{industry}_{value_chain}"
                description = f"{industry}業界の{value_chain}に関する特徴、課題、改善施策をまとめたナレッジ"
        else:  # 能力開発テーマ別
            parts = filename.split('_', 1)
            theme = parts[0]
            industry = parts[1]
            title = f"{theme}_{industry}"
            description = f"{theme}テーマの{industry}業界向けナレッジ"
        
        # タグの整理
        all_tags = []
        for category_tags in tags.values():
            all_tags.extend(category_tags)
        all_tags = list(set(all_tags))  # 重複除去
        
        # ナレッジの内容を作成
        knowledge_content = f"""# {title}

**作成日**: {datetime.now().strftime('%Y年%m月%d日')}

## 🔗 引用元・参照
- [[INBOX/{source_file.name}]]
- 担当者: {participants if participants else '不明'}

**タグ**: {' '.join([f'#{tag}' for tag in all_tags])}

## 📋 概要

{description}

## 🏢 業界・組織特徴

### 業界の特徴
- **業界特有の働き方**: 
- **市場感**: 
- **人材構成**: 
- **コミュニケーション文化**: 
- **よくある問題**: 

### 組織特徴
- **規模別の特徴**: 
- **働き方**: 
- **人材構成**: 
- **コミュニケーション**: 
- **よくある問題**: 

## 🎯 問題・課題

### 現状の課題
（元ファイルから自動抽出予定）

### 問題の背景・要因
（元ファイルから自動抽出予定）

## 💡 打ち手・施策

### 具体的な施策
（元ファイルから自動抽出予定）

### 実施方法
- **ステップ**: 
- **期間**: 
- **予算**: 
- **体制**: 

### 成功・失敗要因
（元ファイルから自動抽出予定）

## 📊 結果・効果

### 効果・成果
（元ファイルから自動抽出予定）

### 他社事例との比較
（類似業界・規模での事例を検索予定）

## 🔗 関連ナレッジ・知見

### 類似業界・規模での事例
（自動検索・関連付け予定）

### 知っておくべき知識
（業界標準・法的要件・ベストプラクティス）

## 📝 備考・考察
（元ファイルから自動抽出予定）

---
**最終更新**: {datetime.now().strftime('%Y年%m月%d日')}
**作成者**: ナレッジ階層化システム
"""
        
        return knowledge_content

    def process_inbox(self):
        """INBOXフォルダのファイルを処理"""
        if not self.inbox_path.exists():
            print(f"INBOXフォルダが見つかりません: {self.inbox_path}")
            return
        
        for file_path in self.inbox_path.glob("*.md"):
            print(f"処理中: {file_path.name}")
            
            # タグを抽出
            tags = self.analyze_file(file_path)
            
            # ナレッジファイルの種類と配置場所を決定
            knowledge_files = self.determine_knowledge_files(file_path, tags)
            
            # 既存のタグ行を確認
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # タグが既に存在しない場合のみ追加
            if '**タグ**:' not in content:
                # タグをファイルに追加
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('**参加者**:'):
                        # 参加者の行の後にタグを挿入
                        all_tags = []
                        for category_tags in tags.values():
                            all_tags.extend(category_tags)
                        all_tags = list(set(all_tags))  # 重複除去
                        
                        lines.insert(i + 2, f'\n**タグ**: {" ".join([f"#{tag}" for tag in all_tags])}')
                        break
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            
            # ナレッジファイルを作成
            for folder_type, filename in knowledge_files:
                if folder_type == '業界別':
                    target_path = self.industry_path / f"{filename}.md"
                else:
                    target_path = self.theme_path / f"{filename}.md"
                
                if not target_path.exists():
                    knowledge_content = self.create_knowledge_content(folder_type, filename, file_path, tags)
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(knowledge_content)
                    print(f"  作成: {folder_type}/{filename}.md")
            
            print(f"  タグ: {tags}")

    def show_tag_statistics(self):
        """タグ統計を表示"""
        stats = self.get_tag_statistics()
        print("\n=== タグ統計 ===")
        for category, count in stats.items():
            print(f"{category}: {count}個のタグ")
        print(f"総タグ数: {len(self.get_all_tags())}個")

    def run(self):
        """システムを実行"""
        print("ナレッジ階層化システムを開始します...")
        print("※ タグは動的に追加・更新可能です（拡張性重視設計）")
        
        # KNOWLEDGEフォルダとサブフォルダを作成
        self.knowledge_path.mkdir(exist_ok=True)
        self.industry_path.mkdir(exist_ok=True)
        self.theme_path.mkdir(exist_ok=True)
        
        # タグ統計を表示
        self.show_tag_statistics()
        
        # INBOXを処理
        print("\nINBOXファイルを処理中...")
        self.process_inbox()
        
        print("\n処理が完了しました！")
        print("新しいタグを追加する場合は add_tag_rule() メソッドを使用してください。")

def main():
    # Obsidian Vaultのパスを指定
    vault_path = "/Users/satonodoka/Documents/Obsidian Vault"
    
    # システムを初期化して実行
    system = KnowledgeHierarchySystem(vault_path)
    system.run()

if __name__ == "__main__":
    main() 