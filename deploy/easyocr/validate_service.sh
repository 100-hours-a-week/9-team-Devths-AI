#!/bin/bash

# EasyOCR Container Health Validation
# Note: In docker_deploy.sh, it maps internal port 8000 to host port 8002.
for i in {1..6}; do
    sleep 5
    if curl -s "http://localhost:8002/health" > /dev/null; then
        echo "✅ EasyOCR Service is healthy!"
        exit 0
    fi
done

echo "❌ EasyOCR Service validation failed!"
exit 1
