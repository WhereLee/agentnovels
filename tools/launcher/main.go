package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	// 获取 exe 所在目录
	exePath, err := os.Executable()
	if err != nil {
		fmt.Printf("获取路径失败：%v\n", err)
		waitExit()
		return
	}
	exeDir := filepath.Dir(exePath)
	scriptPath := filepath.Join(exeDir, "main.py")

	// 检查 main.py 是否存在
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		fmt.Println("错误：未找到 main.py")
		fmt.Printf("请确保本程序与 main.py 在同一目录下。\n")
		fmt.Printf("当前目录：%s\n", exeDir)
		waitExit()
		return
	}

	// 查找 Python
	pythonCmd := findPython()
	if pythonCmd == "" {
		fmt.Println("错误：未找到 Python，请先安装 Python 3.10+。")
		waitExit()
		return
	}

	// 运行 main.py（无参数 → 自动进入聊天模式）
	cmd := exec.Command(pythonCmd, scriptPath)
	cmd.Dir = exeDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Printf("\n程序异常退出：%v\n", err)
	}

	waitExit()
}

func findPython() string {
	candidates := []string{"python", "python3", "py"}
	for _, c := range candidates {
		if path, err := exec.LookPath(c); err == nil {
			return path
		}
	}
	return ""
}

func waitExit() {
	fmt.Print("\n按回车键退出...")
	bufio.NewReader(os.Stdin).ReadBytes('\n')
}
