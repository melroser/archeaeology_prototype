#!/bin/bash
set -e

# Build frontend
echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Copy build to backend
cp -r frontend/build ./src/frontend_build

# Start backend
echo "🚀 Starting FastAPI server..."
cd src
uvicorn archaeologist.web_ui:app --reload
