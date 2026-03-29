#!/bin/bash
cd /vercel/share/v0-project

echo "=== Git 状态检查 ==="
git status

echo ""
echo "=== 检查新增文件 ==="
git status --short

echo ""
echo "=== 添加所有新文件和修改 ==="
git add -A

echo ""
echo "=== 最终状态 ==="
git status --short
