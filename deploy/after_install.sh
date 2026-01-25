#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Installing dependencies with Poetry..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
cd "$APP_DIR"

# 1. Python 버전 확인
echo "🐍 Checking Python version..."
python3 --version

# 2. Poetry 설치 확인 및 설치
if ! command -v poetry &> /dev/null; then
    echo "📥 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 - --quiet
    export PATH="/home/ubuntu/.local/bin:$PATH"

    # Poetry를 시스템 PATH에 영구 추가
    if ! grep -q "poetry" /home/ubuntu/.bashrc; then
        echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> /home/ubuntu/.bashrc
    fi
else
    echo "✅ Poetry already installed"
fi

# Poetry 버전 확인
poetry --version

# 3. Poetry 설정 (가상환경을 프로젝트 내부에 생성)
echo "⚙️  Configuring Poetry..."
poetry config virtualenvs.in-project true

# 4. 의존성 설치 (production 환경이므로 dev 제외)
echo "📚 Installing Python dependencies..."
if [ -f "pyproject.toml" ] && [ -f "poetry.lock" ]; then
    poetry install --only main --no-interaction --no-ansi
    echo "✅ Dependencies installed successfully"
else
    echo "❌ pyproject.toml or poetry.lock not found!"
    exit 1
fi

# 5. 파일 권한 설정
echo "🔐 Setting file permissions..."
chown -R ubuntu:ubuntu "$APP_DIR"
chmod +x "$APP_DIR/deploy/"*.sh

# 6. Import 검증
echo "🧪 Validating Python imports..."
if poetry run python -c "import app.main; print('✅ Main app imports successfully')" 2>/dev/null; then
    echo "✅ Import validation passed"
else
    echo "⚠️  Import validation failed, but continuing..."
fi

echo "✅ After-install steps completed"
exit 0
