import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// オブシディアンファイルの検索API
export async function POST(request: NextRequest) {
  try {
    const { query } = await request.json()
    
    if (!query || typeof query !== 'string') {
      return NextResponse.json({ error: 'クエリが必要です' }, { status: 400 })
    }

    // オブシディアンファイルを検索
    const searchResult = await searchObsidianFiles(query)
    
    // searchResultがnullでない場合、その構造を展開してレスポンスに含める
    if (searchResult && searchResult.result) {
      return NextResponse.json({
        success: true,
        result: searchResult.result,
        totalMatches: searchResult.totalMatches || 0,
        returned: searchResult.returned || 0,
        query: query,
        debugInfo: searchResult.debugInfo
      })
    } else {
      return NextResponse.json({
        success: true,
        result: null,
        totalMatches: 0,
        returned: 0,
        query: query,
        debugInfo: searchResult?.debugInfo
      })
    }
  } catch (error) {
    console.error('検索エラー:', error)
    return NextResponse.json({ 
      error: 'ファイル検索中にエラーが発生しました',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

// オブシディアンファイルを検索する関数
async function searchObsidianFiles(query: string): Promise<any> {
  // パスを明示的に設定
  const obsidianBasePath = path.resolve(process.cwd(), '..')
  console.log('検索クエリ:', query)
  console.log('process.cwd():', process.cwd())
  console.log('オブシディアンベースパス:', obsidianBasePath)
  console.log('ベースパス存在確認:', fs.existsSync(obsidianBasePath))
  
  // 検索パスを明示的に設定
  const searchPaths = [
    path.resolve(obsidianBasePath, 'ナレッジ'),
    path.resolve(obsidianBasePath, 'ファクト'),
    path.resolve(obsidianBasePath, '提案')
  ]

  const matchingFiles: any[] = []
  let totalFilesFound = 0
  const debugInfo = {
    cwd: process.cwd(),
    obsidianBasePath,
    searchPaths,
    pathExistence: {} as Record<string, boolean>,
    filesFound: [] as string[],
    matchResults: [] as any[]
  }

  for (const searchPath of searchPaths) {
    const pathExists = fs.existsSync(searchPath)
    console.log('検索パス確認:', searchPath, '存在:', pathExists)
    debugInfo.pathExistence[searchPath] = pathExists
    
    if (fs.existsSync(searchPath)) {
      const files = await getAllMarkdownFiles(searchPath)
      totalFilesFound += files.length
      console.log(`${searchPath} 内のファイル数:`, files.length)
      
      for (const filePath of files) {
        try {
          console.log('ファイル読み込み試行:', filePath)
          debugInfo.filesFound.push(filePath)
          
          const content = fs.readFileSync(filePath, { encoding: 'utf8' })
          const fileInfo = analyzeMarkdownFile(content, filePath)
          console.log('ファイル解析結果:', {
            fileName: fileInfo.fileName,
            tags: fileInfo.tags,
            contentLength: fileInfo.content.length,
            firstLine: fileInfo.content.substring(0, 50) + '...'
          })
          
          // キーワードがタグまたは内容に含まれているかチェック
          const matches = matchesQuery(fileInfo, query)
          console.log(`${fileInfo.fileName} マッチ結果:`, matches)
          console.log(`クエリ "${query}" vs ファイル名 "${fileInfo.fileName.toLowerCase()}"`)
          
          debugInfo.matchResults.push({
            fileName: fileInfo.fileName,
            filePath,
            matches,
            tags: fileInfo.tags,
            contentLength: fileInfo.content.length
          })
          
          if (matches) {
            // 関連度スコアを計算
            const relevanceScore = calculateRelevanceScore(fileInfo, query)
            matchingFiles.push({ ...fileInfo, relevanceScore })
            console.log('マッチしたファイルを追加:', fileInfo.fileName, 'スコア:', relevanceScore)
          }
        } catch (error) {
          console.error(`ファイル読み込みエラー: ${filePath}`, error)
          debugInfo.matchResults.push({
            fileName: 'ERROR',
            filePath,
            matches: false,
            error: error instanceof Error ? error.message : 'Unknown error'
          })
        }
      }
    }
  }
  
  console.log('総ファイル数:', totalFilesFound, 'マッチファイル数:', matchingFiles.length)
  console.log('マッチしたファイル一覧:', matchingFiles.map(f => f.fileName))

  // デバッグ情報を追加
  debugInfo.filesFound = debugInfo.filesFound || []
  debugInfo.matchResults = debugInfo.matchResults || []

  // 関連度スコア順にソートして最大3件を返す
  if (matchingFiles.length > 0) {
    const sortedFiles = matchingFiles
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 3)
    
    console.log('選択されたファイル:', sortedFiles.map(f => `${f.fileName}(${f.relevanceScore})`))
    return { 
      result: sortedFiles,
      debugInfo,
      totalMatches: matchingFiles.length,
      returned: sortedFiles.length
    }
  }

  console.log('マッチするファイルが見つかりませんでした')
  return { result: null, debugInfo }
}

// ディレクトリ内のすべてのマークダウンファイルを再帰的に取得
async function getAllMarkdownFiles(dirPath: string): Promise<string[]> {
  const files: string[] = []
  
  try {
    // UTF-8エンコーディングを明示的に指定
    const entries = fs.readdirSync(dirPath, { withFileTypes: true, encoding: 'utf8' })
    console.log(`ディレクトリ読み込み: ${dirPath}`)
    console.log(`エントリ数: ${entries.length}`)
    
    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name)
      console.log(`エントリ: ${entry.name} (${entry.isDirectory() ? 'dir' : 'file'})`)
      
      if (entry.isDirectory()) {
        // 再帰的にサブディレクトリを検索
        const subFiles = await getAllMarkdownFiles(fullPath)
        files.push(...subFiles)
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        console.log(`マークダウンファイル追加: ${fullPath}`)
        files.push(fullPath)
      }
    }
  } catch (error) {
    console.error(`ディレクトリ読み込みエラー: ${dirPath}`, error)
  }
  
  return files
}

// マークダウンファイルを解析する関数
function analyzeMarkdownFile(content: string, filePath: string) {
  // path.basename を使わずに、パスを直接処理してUTF-8文字化けを回避
  const fileName = filePath.split(path.sep).pop()?.replace('.md', '') || 'ファイル名不明'
  const tags = extractTags(content)
  
  // 文字エンコーディングのデバッグ情報を追加
  console.log(`ファイル名: "${fileName}" (長さ: ${fileName.length})`)
  console.log(`内容の最初の100文字: "${content.substring(0, 100)}"`)
  
  // summaryの改善: 最初の見出しまたは内容の冒頭を使用
  let summary = '概要なし'
  const lines = content.split('\n')
  
  // 最初の#見出しを探す
  const headingLine = lines.find(line => line.trim().startsWith('# '))
  if (headingLine) {
    summary = headingLine.replace(/^#+\s*/, '').substring(0, 100)
  } else if (content.length > 0) {
    // 見出しがない場合は最初の数行を使用
    summary = content.substring(0, 150).replace(/\n/g, ' ').trim()
    if (content.length > 150) summary += '...'
  }
  
  return {
    fileName,
    filePath,
    content,
    tags,
    summary
  }
}

// タグを抽出する関数
function extractTags(content: string): string[] {
  const tags: string[] = []
  
  // #タグ形式
  const hashTags = content.match(/#[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+/g)
  if (hashTags) {
    tags.push(...hashTags.map(tag => tag.substring(1)))
  }
  
  // [[タグ]]形式
  const bracketTags = content.match(/\[\[([^\]]+)\]\]/g)
  if (bracketTags) {
    tags.push(...bracketTags.map(tag => tag.slice(2, -2)))
  }
  
  // YAML frontmatter内のタグ
  const yamlMatch = content.match(/^---\n([\s\S]*?)\n---/)
  if (yamlMatch) {
    const yamlContent = yamlMatch[1]
    const tagsMatch = yamlContent.match(/tags:\s*\[(.*?)\]/)
    if (tagsMatch) {
      const yamlTags = tagsMatch[1].split(',').map(tag => tag.trim().replace(/['"]/g, ''))
      tags.push(...yamlTags)
    }
  }
  
  return Array.from(new Set(tags)) // 重複を除去
}

// 関連度スコアを計算する関数
function calculateRelevanceScore(fileInfo: any, query: string): number {
  const queryLower = query.toLowerCase()
  let score = 0
  
  // ファイル名での一致（高得点）
  const fileNameLower = fileInfo.fileName.toLowerCase()
  if (fileNameLower.includes(queryLower)) {
    score += 50
    // 完全一致の場合はボーナス
    if (fileNameLower === queryLower) {
      score += 30
    }
  }
  
  // タグでの一致（中得点）
  for (const tag of fileInfo.tags) {
    const tagLower = tag.toLowerCase()
    if (tagLower.includes(queryLower)) {
      score += 30
      // 完全一致の場合はボーナス
      if (tagLower === queryLower) {
        score += 20
      }
    }
  }
  
  // 内容での一致（低得点、但し出現回数に応じて加算）
  const contentLower = fileInfo.content.toLowerCase()
  const matches = (contentLower.match(new RegExp(queryLower, 'g')) || []).length
  score += matches * 2
  
  // ファイルの長さで正規化（短いファイルほど関連度が高いと仮定）
  if (fileInfo.content.length < 1000) {
    score += 10
  } else if (fileInfo.content.length < 3000) {
    score += 5
  }
  
  return Math.round(score)
}

// クエリがファイルにマッチするかチェック
function matchesQuery(fileInfo: any, query: string): boolean {
  const queryLower = query.toLowerCase()
  console.log(`マッチング処理開始: クエリ="${query}" (小文字: "${queryLower}")`)
  
  // ファイル名での検索
  const fileNameLower = fileInfo.fileName.toLowerCase()
  console.log(`ファイル名チェック: "${fileNameLower}" に "${queryLower}" が含まれるか`)
  if (fileNameLower.includes(queryLower)) {
    console.log('ファイル名でマッチ!')
    return true
  }
  
  // タグでの検索
  console.log(`タグチェック: ${fileInfo.tags.length}個のタグ:`, fileInfo.tags)
  if (fileInfo.tags.some((tag: string) => {
    const tagLower = tag.toLowerCase()
    console.log(`タグ "${tagLower}" に "${queryLower}" が含まれるか`)
    return tagLower.includes(queryLower)
  })) {
    console.log('タグでマッチ!')
    return true
  }
  
  // 内容での検索
  const contentLower = fileInfo.content.toLowerCase()
  console.log(`内容チェック: ${fileInfo.content.length}文字の内容に "${queryLower}" が含まれるか`)
  if (contentLower.includes(queryLower)) {
    console.log('内容でマッチ!')
    return true
  }
  
  console.log('マッチしませんでした')
  return false
} 