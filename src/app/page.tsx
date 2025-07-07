'use client'

import { useState, useEffect } from 'react'

interface Message {
  id: number
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  knowledgeCards?: KnowledgeCard[]
  step?: number
  stepType?: 'initial' | 'deep-dive' | 'proposal'
}

interface KnowledgeCard {
  id: string
  title: string
  summary: string
  tags: string[]
  relevance: number
}

interface ChatHistory {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState('')
  const [isEditingChatName, setIsEditingChatName] = useState<string | null>(null)
  const [editingChatName, setEditingChatName] = useState('')
  const [currentStep, setCurrentStep] = useState(1)
  const [currentStepType, setCurrentStepType] = useState<'initial' | 'deep-dive' | 'proposal'>('initial')

  const mockKnowledgeCards: KnowledgeCard[] = [
    {
      id: '1',
      title: '建設業界における女性活躍推進の成功事例',
      summary: '五洋建設では女性管理職比率を5年間で15%向上させることに成功。メンタープログラムとフレックス制度の導入が効果的でした。',
      tags: ['建設業界', '女性活躍', '管理職', 'メンター制度'],
      relevance: 95
    },
    {
      id: '2',
      title: 'IT業界の人材定着率改善施策',
      summary: '日本電子計算では新人研修プログラムの見直しとキャリアパス明確化により、新卒3年後離職率を40%から15%に改善。',
      tags: ['IT業界', '人材定着', '新人研修', 'キャリアパス'],
      relevance: 87
    },
    {
      id: '3',
      title: '評価制度改革による組織活性化',
      summary: '目標設定の透明性向上と360度評価の導入により、従業員満足度が30%向上。特に中間管理職の評価が改善されました。',
      tags: ['評価制度', '組織改革', '360度評価', '従業員満足度'],
      relevance: 82
    }
  ]

  // ローカルストレージからチャット履歴を読み込み
  useEffect(() => {
    const savedHistory = localStorage.getItem('chatHistory')
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory).map((chat: any) => ({
          ...chat,
          createdAt: new Date(chat.createdAt),
          updatedAt: new Date(chat.updatedAt),
          messages: chat.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }))
        setChatHistory(parsedHistory)
      } catch (error) {
        console.error('チャット履歴の読み込みに失敗:', error)
      }
    }
  }, [])

  // 検索クエリのデバウンス処理
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchQuery(searchQuery)
    }, 300)
    
    return () => clearTimeout(timer)
  }, [searchQuery])

  // チャット履歴をローカルストレージに保存
  const saveToLocalStorage = (history: ChatHistory[]) => {
    try {
      localStorage.setItem('chatHistory', JSON.stringify(history))
    } catch (error) {
      console.error('ローカルストレージへの保存に失敗:', error)
      // 容量制限の場合は古いデータを削除
      if (history.length > 10) {
        const reducedHistory = history.slice(0, 10)
        try {
          localStorage.setItem('chatHistory', JSON.stringify(reducedHistory))
          setChatHistory(reducedHistory)
        } catch (retryError) {
          console.error('データ削減後も保存に失敗:', retryError)
        }
      }
    }
  }

  // 新しいチャットを作成
  const createNewChat = () => {
    const newChat: ChatHistory = {
      id: Date.now().toString(),
      title: '新しいチャット',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    }
    const newHistory = [newChat, ...chatHistory]
    setChatHistory(newHistory)
    setCurrentChatId(newChat.id)
    setMessages([])
    setCurrentStep(1)
    setCurrentStepType('initial')
    saveToLocalStorage(newHistory)
  }

  // チャットを削除
  const deleteChat = (chatId: string) => {
    const newHistory = chatHistory.filter(chat => chat.id !== chatId)
    setChatHistory(newHistory)
    if (currentChatId === chatId) {
      setCurrentChatId(null)
      setMessages([])
    }
    saveToLocalStorage(newHistory)
  }

  // チャット名を変更
  const renameChatTitle = (chatId: string, newTitle: string) => {
    const newHistory = chatHistory.map(chat => 
      chat.id === chatId 
        ? { ...chat, title: newTitle, updatedAt: new Date() }
        : chat
    )
    setChatHistory(newHistory)
    saveToLocalStorage(newHistory)
    setIsEditingChatName(null)
    setEditingChatName('')
  }

  // チャットを選択
  const selectChat = (chatId: string) => {
    const selectedChat = chatHistory.find(chat => chat.id === chatId)
    if (selectedChat) {
      setCurrentChatId(chatId)
      setMessages(selectedChat.messages)
      // 最新メッセージからステップ情報を復元
      const lastAssistantMessage = selectedChat.messages
        .filter(msg => msg.type === 'assistant')
        .pop()
      if (lastAssistantMessage && lastAssistantMessage.step) {
        setCurrentStep(lastAssistantMessage.step + 1)
        setCurrentStepType(lastAssistantMessage.stepType || 'initial')
      } else {
        setCurrentStep(1)
        setCurrentStepType('initial')
      }
    }
  }

  // チャット履歴を検索
  const filteredChatHistory = chatHistory.filter(chat =>
    chat.title.toLowerCase().includes(debouncedSearchQuery.toLowerCase()) ||
    chat.messages.some(msg => 
      msg.content.toLowerCase().includes(debouncedSearchQuery.toLowerCase())
    )
  )

  // 現在のチャットの更新
  const updateCurrentChat = (newMessages: Message[]) => {
    if (currentChatId) {
      const newHistory = chatHistory.map(chat => 
        chat.id === currentChatId 
          ? { ...chat, messages: newMessages, updatedAt: new Date() }
          : chat
      )
      setChatHistory(newHistory)
      saveToLocalStorage(newHistory)
    }
  }

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    // 現在のチャットがない場合は新しいチャットを作成
    let chatId = currentChatId
    if (!chatId) {
      const newChat: ChatHistory = {
        id: Date.now().toString(),
        title: inputValue.length > 30 ? inputValue.substring(0, 30) + '...' : inputValue,
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      }
      const newHistory = [newChat, ...chatHistory]
      setChatHistory(newHistory)
      setCurrentChatId(newChat.id)
      saveToLocalStorage(newHistory)
      chatId = newChat.id
    }

    const userMessage: Message = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInputValue('')
    setIsLoading(true)
    updateCurrentChat(newMessages)

    try {
      // Chat APIに送信
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages.map(msg => ({
          role: msg.type === 'user' ? 'user' : 'assistant',
          content: msg.content
        })) })
      })

      if (!response.ok) {
        throw new Error(`Chat API error: ${response.status}`)
      }

      const data = await response.json()
      
      if (!data.success) {
        throw new Error(data.error || 'APIエラーが発生しました')
      }

      const assistantMessage: Message = {
        id: Date.now() + 1,
        type: 'assistant',
        content: data.message,
        timestamp: new Date(),
        knowledgeCards: mockKnowledgeCards.filter(card => data.message.includes(card.title)),
        step: currentStep,
        stepType: currentStepType
      }

      const finalMessages = [...newMessages, assistantMessage]
      setMessages(finalMessages)
      updateCurrentChat(finalMessages)
      setCurrentStep(currentStep + 1)
      setCurrentStepType(currentStepType === 'initial' ? 'deep-dive' : 'proposal')
      setIsLoading(false)
    } catch (error) {
      console.error('Chat API呼び出しエラー:', error)
      // エラー時はフォールバック応答
      const assistantMessage: Message = {
        id: Date.now() + 1,
        type: 'assistant',
        content: '申し訳ございません。チャットの処理中にエラーが発生しました。もう一度お試しください。',
        timestamp: new Date(),
        step: currentStep,
        stepType: currentStepType
      }
      const finalMessages = [...newMessages, assistantMessage]
      setMessages(finalMessages)
      updateCurrentChat(finalMessages)
      setIsLoading(false)
    }
  }

  // テキストハイライト機能
  const highlightText = (text: string, query: string) => {
    if (!query) return text
    const parts = text.split(new RegExp(`(${query})`, 'gi'))
    return parts.map((part, index) => 
      part.toLowerCase() === query.toLowerCase() ? (
        <span key={index} className="bg-yellow-200 text-black px-1 rounded">
          {part}
        </span>
      ) : (
        part
      )
    )
  }

  // 段階的対話のためのレスポンス生成関数
  const generateStepBasedResponse = (userQuery: string, step: number, messageHistory: Message[]): string => {
    switch (step) {
      case 1:
        // 初回質問：3つの基本質問のみ（ナレッジカードなし）
        return `「${userQuery}」についてお聞かせください。

より具体的で有用な提案をするために、以下の点について教えてください：

1. **どちらの業界・分野でしょうか？**
   （例：建設業、IT、製造業、サービス業など）

2. **企業規模はどの程度ですか？**
   （例：大手企業、中堅企業、中小企業、スタートアップなど）

3. **現在どのような課題を感じていますか？**
   （例：人材不足、離職率、組織風土、評価制度など）

これらの情報をお聞かせいただければ、より的確なアドバイスができます。`

      case 2:
        // 2回目：業界情報に基づく深掘り質問
        return `ありがとうございます。「${userQuery}」ですね。

その業界でよくある課題について、もう少し詳しく教えてください：

- 具体的にはどのような問題が一番深刻でしょうか？
- 現在何か対策は取られていますか？
- 解決の優先度はどの程度でしょうか？

お聞かせいただければ、より具体的なアドバイスができます。`

      case 3:
        // 3回目：さらに詳しい情報収集
        return `詳しい情報をありがとうございます。「${userQuery}」について理解が深まりました。

最後に、以下についてお聞かせください：

- 取り組み開始時期の目安はありますか？
- 予算や人員の制約はありますか？
- 成功の基準や目標はどのようなものでしょうか？

これらの情報が揃いましたら、最適なナレッジとソリューションをご提案いたします。`

      default:
        // 4回目以降：基本的な確認
        return `ありがとうございます。これまでの情報を整理して、最適なナレッジを検索いたします。

少々お待ちください...`
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* サイドバー */}
      <div className="w-64 bg-gray-900 text-white flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-xl font-bold">AIナレッジチャット</h1>
        </div>
        
        <div className="p-4">
          <button 
            onClick={createNewChat}
            className="w-full bg-gray-800 hover:bg-gray-700 text-white py-2 px-4 rounded-lg text-left"
          >
            + 新しいチャット
          </button>
        </div>
        
        {/* 検索機能 */}
        <div className="px-4 pb-4">
          <input
            type="text"
            placeholder="チャットを検索..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-gray-800 text-white px-3 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <div className="flex-1 overflow-y-auto">
          <div className="p-4 space-y-2">
            {filteredChatHistory.length > 0 ? (
              <>
                <div className="text-sm text-gray-300">チャット履歴</div>
                <div className="space-y-1">
                  {filteredChatHistory.map((chat) => (
                    <div
                      key={chat.id}
                      className={`group flex items-center justify-between px-3 py-2 text-sm rounded cursor-pointer hover:bg-gray-800 ${
                        currentChatId === chat.id ? 'bg-gray-700' : ''
                      }`}
                      onClick={() => selectChat(chat.id)}
                    >
                      {isEditingChatName === chat.id ? (
                        <input
                          type="text"
                          value={editingChatName}
                          onChange={(e) => setEditingChatName(e.target.value)}
                          onBlur={() => renameChatTitle(chat.id, editingChatName)}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              renameChatTitle(chat.id, editingChatName)
                            }
                          }}
                          className="bg-gray-800 text-white px-2 py-1 rounded text-sm w-full focus:outline-none"
                          autoFocus
                        />
                      ) : (
                                                 <>
                           <span className="flex-1 truncate">
                             {highlightText(chat.title, debouncedSearchQuery)}
                           </span>
                           <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100">
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                setIsEditingChatName(chat.id)
                                setEditingChatName(chat.title)
                              }}
                              className="text-gray-400 hover:text-white p-1"
                            >
                              ✏️
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                deleteChat(chat.id)
                              }}
                              className="text-gray-400 hover:text-red-400 p-1"
                            >
                              🗑️
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </>
                           ) : debouncedSearchQuery ? (
              <div className="text-sm text-gray-500">検索結果が見つかりません</div>
            ) : (
              <div className="text-sm text-gray-500">まだチャットがありません</div>
            )}
          </div>
        </div>
      </div>

      {/* メインコンテンツ */}
      <div className="flex-1 flex flex-col">
        {/* チャットエリア */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  のどか さん、お久しぶりです。
                </h2>
                <p className="text-gray-600 mb-8">
                  ナレッジに関する質問をお聞かせください
                </p>
                <div className="bg-gray-50 rounded-lg p-6 max-w-2xl">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">
                    こんなふうに話しかけてください
                  </h3>
                  <div className="space-y-3">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="text-gray-800 font-medium">「建設業界での女性活躍推進の成功事例を教えて」</div>
                      <div className="text-sm text-gray-600 mt-1">→ 具体的な業界と課題を指定した質問</div>
                    </div>
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="text-gray-800 font-medium">「新卒の離職率を下げるには何が効果的？」</div>
                      <div className="text-sm text-gray-600 mt-1">→ 解決策を求める実務的な質問</div>
                    </div>
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="text-gray-800 font-medium">「360度評価制度のメリット・デメリットは？」</div>
                      <div className="text-sm text-gray-600 mt-1">→ 特定の制度について詳しく知りたい質問</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 max-w-3xl mx-auto">
              {messages.map((message) => (
                <div key={message.id} className="space-y-4">
                  <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] p-4 rounded-lg ${
                      message.type === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-white border border-gray-200'
                    }`}>
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      <div className={`text-xs mt-2 flex items-center justify-between ${
                        message.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                      }`}>
                        <span>
                          {message.timestamp.toLocaleTimeString('ja-JP', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </span>
                        {message.step && message.type === 'assistant' && (
                          <span className="px-2 py-1 rounded text-xs bg-gray-100 text-gray-600">
                            STEP {message.step}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {/* ナレッジカード */}
                  {message.knowledgeCards && message.knowledgeCards.length > 0 && (
                    <div className="ml-4 space-y-3">
                      <div className="text-sm text-gray-600 font-medium">関連ナレッジ</div>
                      {message.knowledgeCards.map((card) => (
                        <div key={card.id} className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <div className="flex items-start justify-between mb-2">
                            <h4 className="font-medium text-gray-900 text-sm">{card.title}</h4>
                            <div className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                              {card.relevance}% 関連
                            </div>
                          </div>
                          <p className="text-sm text-gray-700 mb-3">{card.summary}</p>
                          <div className="flex flex-wrap gap-1">
                            {card.tags.map((tag, index) => (
                              <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span className="text-gray-600">回答を生成中...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 入力エリア */}
        <div className="border-t border-gray-200 p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end space-x-2">
              <div className="flex-1">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="質問してみましょう"
                  className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={3}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                送信
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 