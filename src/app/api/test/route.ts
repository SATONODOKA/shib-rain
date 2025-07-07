import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
  try {
    const obsidianBasePath = path.resolve(process.cwd(), '..')
    const knowledgePath = path.resolve(obsidianBasePath, 'ナレッジ')
    
    console.log('process.cwd():', process.cwd())
    console.log('obsidianBasePath:', obsidianBasePath)
    console.log('knowledgePath:', knowledgePath)
    
    const result = {
      cwd: process.cwd(),
      obsidianBasePath: obsidianBasePath,
      knowledgePath: knowledgePath,
      obsidianExists: fs.existsSync(obsidianBasePath),
      knowledgeExists: fs.existsSync(knowledgePath),
      knowledgeFiles: [] as Array<{ name: string; fullPath: string }>
    }
    
    if (fs.existsSync(knowledgePath)) {
      try {
        const files = fs.readdirSync(knowledgePath)
        result.knowledgeFiles = files.filter(f => f.endsWith('.md')).map(f => ({
          name: f,
          fullPath: path.join(knowledgePath, f)
        }))
      } catch (error) {
        console.error('ディレクトリ読み込みエラー:', error)
      }
    }
    
    return NextResponse.json(result)
  } catch (error) {
    console.error('テストエラー:', error)
    return NextResponse.json({ 
      error: 'テスト中にエラーが発生しました',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
} 