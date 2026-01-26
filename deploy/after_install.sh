#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Installing dependencies with Poetry..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
cd "$APP_DIR"

# 0. 시스템 패키지 설치 (ML dependencies 필수)
echo "📦 Installing system packages for ML dependencies..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ System packages installed"
    tesseract --version | head -n 1
else
    echo "⚠️  System package installation failed, but continuing..."
fi
echo ""

# 1. Python 버전 확인 및 pyenv 설정
echo "🐍 Checking Python version..."

# pyenv 경로 설정 (존재하는 경우)
if [ -d "/home/ubuntu/.local/share/pyenv" ]; then
    export PYENV_ROOT="/home/ubuntu/.local/share/pyenv"
    export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
    echo "✅ pyenv found, adding to PATH"

    # pyenv 초기화
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)"
        echo "✅ pyenv initialized"
    fi
fi

python3 --version
PYTHON_PATH=$(which python3)

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

# 4. Poetry에 Python 버전 명시적으로 지정
echo "🔧 Setting Python version for Poetry..."
echo "📍 Using Python: $PYTHON_PATH"
poetry env use "$PYTHON_PATH"

# 5. Lock 파일 업데이트 (Python 버전 호환성 보장)
echo "🔄 Updating poetry.lock for current Python version..."
if [ -f "pyproject.toml" ]; then
    # Python 버전 호환성을 위해 lock 파일만 업데이트 (의존성 버전은 유지)
    poetry lock --no-update --no-interaction --no-ansi
    if [ $? -eq 0 ]; then
        echo "✅ poetry.lock updated successfully"
    else
        echo "⚠️  Failed to update poetry.lock, attempting install anyway..."
    fi
else
    echo "❌ pyproject.toml not found!"
    exit 1
fi

# 6. 의존성 설치 (production 환경: main + ml group, dev 제외)
echo "📚 Installing Python dependencies (main + ml groups)..."
if [ -f "pyproject.toml" ] && [ -f "poetry.lock" ]; then
    poetry install --with ml --without dev --no-interaction --no-ansi
    echo "✅ Dependencies installed successfully (Core + ML packages)"
else
    echo "❌ pyproject.toml or poetry.lock not found!"
    exit 1
fi

# 7. 파일 권한 설정
echo "🔐 Setting file permissions..."
chown -R ubuntu:ubuntu "$APP_DIR"
chmod +x "$APP_DIR/deploy/"*.sh

# 8. Import 검증
echo "🧪 Validating Python imports..."
if poetry run python -c "import app.main; print('✅ Main app imports successfully')" 2>/dev/null; then
    echo "✅ Import validation passed"
else
    echo "⚠️  Import validation failed, but continuing..."
fi

# 9. 로그 로테이트 설정 (Cron Job)
echo "🔄 Setting up log rotation..."
ROTATE_SCRIPT="$APP_DIR/deploy/rotate_logs.sh"
CRON_LOG="$APP_DIR/logs/cron-rotate.log"

if [ -f "$ROTATE_SCRIPT" ]; then
    chmod +x "$ROTATE_SCRIPT"

    # Cron job 설정 (매일 새벽 3시)
    CRON_SCHEDULE="0 3 * * *"
    CRON_COMMAND="$CRON_SCHEDULE cd $APP_DIR && /bin/bash $ROTATE_SCRIPT >> $CRON_LOG 2>&1"

    # 기존 crontab에서 rotate_logs 제거 후 추가
    TEMP_CRON=$(mktemp)
    crontab -l > "$TEMP_CRON" 2>/dev/null || true
    sed -i.bak '/rotate_logs.sh/d' "$TEMP_CRON"
    echo "$CRON_COMMAND" >> "$TEMP_CRON"

    if crontab "$TEMP_CRON" 2>/dev/null; then
        echo "✅ Log rotation cron job configured (daily at 3:00 AM)"
    else
        echo "⚠️  Cron job setup failed, but continuing..."
    fi

    rm -f "$TEMP_CRON" "${TEMP_CRON}.bak"
else
    echo "⚠️  rotate_logs.sh not found, skipping log rotation setup"
fi

echo "✅ After-install steps completed"
exit 0
