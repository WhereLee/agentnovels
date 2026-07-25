package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	mirrorBase = "https://hf-mirror.com"
	repoID     = "BAAI/bge-base-zh-v1.5"
)

// FileEntry 表示 HuggingFace API 返回的文件条目
type FileEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

func main() {
	fmt.Println("==================================================")
	fmt.Println("  小说问答 AI Agent - Embedding 模型下载工具")
	fmt.Println("==================================================")
	fmt.Printf("\n模型：%s\n", repoID)
	fmt.Printf("镜像：%s\n", mirrorBase)

	// 获取 exe 所在目录，模型存到上级目录的 models/ 下
	exePath, err := os.Executable()
	if err != nil {
		fmt.Printf("获取路径失败：%v\n", err)
		waitExit()
		return
	}
	exeDir := filepath.Dir(exePath)
	modelDir := filepath.Join(exeDir, "..", "models", "bge-base-zh-v1.5")
	modelDir, _ = filepath.Abs(modelDir)

	fmt.Printf("下载到：%s\n\n", modelDir)

	// 1. 获取文件列表
	fmt.Println("正在获取文件列表...")
	files, err := listFiles()
	if err != nil {
		fmt.Printf("获取文件列表失败：%v\n", err)
		fmt.Println("请检查网络连接后重试。")
		waitExit()
		return
	}
	fmt.Printf("共 %d 个文件\n\n", len(files))

	// 2. 逐个下载
	successCount := 0
	for i, f := range files {
		fmt.Printf("[%d/%d] %s (%s)\n", i+1, len(files), f.Path, formatSize(f.Size))
		destPath := filepath.Join(modelDir, filepath.FromSlash(f.Path))

		if err := downloadFile(f, destPath); err != nil {
			fmt.Printf("  ✗ 下载失败：%v\n", err)
			fmt.Println("  重新运行可断点续传。")
			waitExit()
			return
		}
		successCount++
	}

	fmt.Println("\n==================================================")
	fmt.Printf("  ✓ 全部完成！(%d/%d)\n", successCount, len(files))
	fmt.Printf("  模型已保存到：%s\n", modelDir)
	fmt.Println("==================================================")
	waitExit()
}

// listFiles 从 HuggingFace API 获取仓库文件列表
func listFiles() ([]FileEntry, error) {
	url := fmt.Sprintf("%s/api/models/%s/tree/main?recursive=true", mirrorBase, repoID)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API 返回状态码 %d", resp.StatusCode)
	}

	var entries []FileEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}

	// 只保留文件（过滤掉目录和 .gitattributes）
	var files []FileEntry
	for _, e := range entries {
		if e.Type == "file" && !strings.Contains(e.Path, ".gitattributes") {
			files = append(files, e)
		}
	}
	return files, nil
}

// downloadFile 下载单个文件，支持断点续传，显示进度
func downloadFile(f FileEntry, destPath string) error {
	// 确保目录存在
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return err
	}

	// 检查是否已下载完成
	if f.Size > 0 {
		if info, err := os.Stat(destPath); err == nil && info.Size() == f.Size {
			fmt.Println("  已存在，跳过。")
			return nil
		}
	}

	// 检查断点续传
	var startByte int64 = 0
	if info, err := os.Stat(destPath); err == nil {
		startByte = info.Size()
	}

	url := fmt.Sprintf("%s/%s/resolve/main/%s", mirrorBase, repoID, f.Path)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// 如果服务器不支持 Range，重新下载
	if startByte > 0 && resp.StatusCode == 200 {
		startByte = 0
	}

	// 打开/创建文件
	flag := os.O_CREATE | os.O_WRONLY
	if startByte > 0 {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	file, err := os.OpenFile(destPath, flag, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	totalSize := f.Size
	if totalSize == 0 && resp.ContentLength > 0 {
		totalSize = resp.ContentLength + startByte
	}

	// 带进度的写入
	writer := &progressWriter{
		total:      totalSize,
		downloaded: startByte,
		startTime:  time.Now(),
	}

	_, err = io.Copy(io.MultiWriter(file, writer), resp.Body)
	if err != nil {
		return err
	}

	fmt.Println() // 进度条换行
	return nil
}

// progressWriter 实现 io.Writer，在终端显示下载进度条
type progressWriter struct {
	total      int64
	downloaded int64
	startTime  time.Time
	lastUpdate time.Time
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n := len(p)
	pw.downloaded += int64(n)

	// 每 100ms 刷新一次，避免刷屏
	now := time.Now()
	if now.Sub(pw.lastUpdate) < 100*time.Millisecond {
		return n, nil
	}
	pw.lastUpdate = now

	pw.render()
	return n, nil
}

func (pw *progressWriter) render() {
	if pw.total <= 0 {
		fmt.Printf("\r  已下载：%s", formatSize(pw.downloaded))
		return
	}

	percent := float64(pw.downloaded) / float64(pw.total) * 100
	barWidth := 30
	filled := int(percent / 100 * float64(barWidth))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	// 计算速度
	elapsed := time.Since(pw.startTime).Seconds()
	speed := float64(pw.downloaded) / elapsed

	fmt.Printf("\r  [%s] %5.1f%%  %s/s", bar, percent, formatSize(int64(speed)))
}

func formatSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func waitExit() {
	fmt.Print("\n按回车键退出...")
	bufio.NewReader(os.Stdin).ReadBytes('\n')
}
