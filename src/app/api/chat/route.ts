import { NextRequest, NextResponse } from 'next/server'

// 営業コンサルタント風の仮レスポンス生成関数
function generateConsultantResponse(userMessage: string, conversationHistory: any[]): string {
  const message = userMessage.toLowerCase()
  
  // 挨拶や初回質問の場合
  if (conversationHistory.length <= 1 || message.includes('はじめ') || message.includes('こんにち')) {
    return `こんにちは！営業第二の脳チャットです。

どのようなご相談でしょうか？より具体的なアドバイスができるよう、以下について教えていただけますか？

**📋 まずは基本情報から**
1. **業界・事業領域** - どちらの業界でお仕事でしょうか？
2. **企業規模** - 従業員数や売上規模の目安
3. **現在のお悩み** - 具体的にどのような課題をお持ちですか？

お気軽にお聞かせください！`
  }

  // 女性活躍に関する質問
  if (message.includes('女性') && (message.includes('活躍') || message.includes('推進'))) {
    return `女性活躍推進についてですね！これは多くの企業が取り組んでいる重要なテーマです。

**🔍 まず現状を整理させてください**

1. **業界特性の確認**
   - 建設業界ですか？IT業界ですか？
   - 技術職・現場職の比率はいかがでしょうか？

2. **現在の課題レベル**
   - 採用段階での女性比率が低い？
   - 管理職登用が進まない？
   - 離職率が高い？

3. **取り組み規模**
   - 全社規模での制度改革をお考えですか？
   - 特定部門から始めたいですか？

業界や企業規模によって効果的なアプローチが変わりますので、詳しく教えていただけますか？`
  }

  // 建設業界の質問
  if (message.includes('建設') || message.includes('工事') || message.includes('現場')) {
    return `建設業界のご相談ですね！

建設業界特有の課題がありますから、業界の特性を踏まえたアドバイスをさせていただきます。

**🏗️ 建設業界でよくあるご相談**
- 現場と本社の情報共有
- 技術継承・人材育成
- 安全管理の徹底
- 女性活躍推進の難しさ
- 海外展開時の人材管理

具体的にはどのような課題でお困りでしょうか？

また、**企業規模**（従業員数やプロジェクト規模）も教えていただけると、より具体的なアドバイスができます。`
  }

  // IT業界の質問
  if (message.includes('it') || message.includes('システム') || message.includes('エンジニア') || message.includes('開発')) {
    return `IT業界のご相談ですね！

技術の変化が激しい業界ですから、人材関連の課題も独特ですよね。

**💻 IT業界でよくあるご相談**
- エンジニアの採用・定着
- スキルアップ支援
- リモートワークの組織運営
- プロジェクト管理の効率化
- 技術とマネジメントのキャリアパス

どちらの領域でお困りでしょうか？

また、**開発規模**（チーム人数やプロダクト規模）も合わせて教えていただけますか？`
  }

  // 人材関連の質問
  if (message.includes('人材') || message.includes('採用') || message.includes('育成') || message.includes('定着')) {
    return `人材に関するご相談ですね！

人材は企業の最も重要な資産ですから、戦略的に取り組む必要があります。

**👥 人材課題の整理**
1. **採用フェーズ**
   - 母集団形成が困難
   - 求める人材の確保が難しい
   - 内定辞退が多い

2. **育成フェーズ**
   - 新人研修の効果が上がらない
   - OJTが属人的
   - スキルアップが進まない

3. **定着フェーズ**
   - 早期離職が多い
   - モチベーション低下
   - キャリアパスが不明確

どのフェーズで特にお困りでしょうか？

また、**対象となる職種**（営業、技術、管理職など）も教えてください。`
  }

  // 感謝や回答の場合
  if (message.includes('ありがとう') || message.includes('参考') || message.includes('理解')) {
    return `お役に立てて良かったです！😊

もし追加でご質問があれば、お気軽にお聞きください。

**💡 次のステップのご提案**
- より詳細な課題の深掘り
- 具体的な解決策の検討
- 実施計画の策定
- 類似事例の確認

どちらか気になる点があれば、ぜひお聞かせくださいね。`
  }

  // その他の質問に対する汎用レスポンス
  return `ご質問ありがとうございます！

より具体的で実用的なアドバイスをするために、もう少し詳しく教えていただけますか？

**🎯 以下の情報があると助かります**
- **業界・事業内容**
- **企業・部門の規模** 
- **具体的な課題や状況**
- **解決したい期間や目標**

どんな小さなことでも構いませんので、お聞かせください！`
}

export async function POST(request: NextRequest) {
  try {
    const { messages } = await request.json()
    
    // 最新のユーザーメッセージを取得
    const userMessage = messages[messages.length - 1]?.content || ''
    
    // 営業コンサルタント風のレスポンスを生成
    const assistantMessage = generateConsultantResponse(userMessage, messages)

    return NextResponse.json({
      success: true,
      message: assistantMessage
    })

  } catch (error) {
    console.error('Chat API Error:', error)
    return NextResponse.json({
      success: false,
      error: 'APIエラーが発生しました',
      message: 'お手数ですが、もう一度お試しください。問題が続く場合は、サポートにお問い合わせください。'
    }, { status: 500 })
  }
} 