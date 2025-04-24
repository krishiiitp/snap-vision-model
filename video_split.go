package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

func splitVideo(videoPath string, chunkDuration int) ([]string, error) {
	chunkPaths := []string{}
	chunkDir := "chunks"
	if err := os.MkdirAll(chunkDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create chunk directory: %w", err)
	}

	chunkPattern := filepath.Join(chunkDir, "chunk%03d.mp4")
	cmd := exec.Command("ffmpeg", "-i", videoPath, "-c", "copy", "-map", "0", "-segment_time", fmt.Sprintf("%d", chunkDuration), "-f", "segment", chunkPattern)

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg split failed: %w", err)
	}

	files, err := filepath.Glob(filepath.Join(chunkDir, "chunk*.mp4"))
	if err != nil {
		return nil, fmt.Errorf("glob failed: %w", err)
	}

	for _, file := range files {
		chunkPaths = append(chunkPaths, file)
	}

	return chunkPaths, nil
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run video_split.go <video_path>")
	}
	videoPath := os.Args[1]
	chunkDuration := 10

	chunkPaths, err := splitVideo(videoPath, chunkDuration)
	if err != nil {
		log.Fatalf("split video failed: %v", err)
	}

	fmt.Println("Created Chunks...")
	for _, chunkPath := range chunkPaths {
		fmt.Println(chunkPath)
	}
}