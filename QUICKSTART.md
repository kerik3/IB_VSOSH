# VVM Platform - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

✅ Python 3.8+  
✅ Node.js 16+  
✅ FFmpeg installed  

## Installation (Fast Track)

### 1. Backend Setup

```bash
cd backend
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python
```

```python
from app import app, db
with app.app_context():
    db.create_all()
exit()
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Start Servers

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

## First Steps

1. **Open browser:** http://localhost:3000

2. **Login as admin:**
   - Username: `admin`
   - Password: `admin123`

3. **Create teacher account:**
   - Go to Register
   - Role: Teacher
   - Complete form

4. **Create student account:**
   - Go to Register
   - Role: Student
   - Complete form

## Testing the System

### Upload a Video (Teacher)

1. Login as teacher
2. Navigate to "Videos"
3. Click "Upload Video"
4. Select a video file
5. Fill in details
6. Upload

### Grant Access (Teacher)

1. Go to "Videos"
2. Click "Manage Access" on your video
3. Select student(s)
4. Click "Grant Access"

### Watch Video (Student)

1. Login as student
2. Go to "Videos"
3. Click "Watch Video"
4. Video plays with your unique watermark!

### Detect Leak (Teacher/Admin)

1. Download the student's watermarked video
2. Go to "Leak Detection"
3. Upload the video
4. Click "Analyze"
5. System identifies the student

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    VVM Platform                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Frontend (React)          Backend (Flask)               │
│  ├─ Login/Register         ├─ Authentication             │
│  ├─ Dashboard              ├─ Video Management           │
│  ├─ Video Player           ├─ Watermarking Engine        │
│  ├─ Upload Interface       ├─ Leak Detection             │
│  └─ Admin Panel            └─ User Management            │
│                                                           │
│  Watermarking System                                      │
│  ├─ Video Watermark (DCT-based)                          │
│  ├─ Audio Watermark (FFT-based)                          │
│  └─ Forensic Extraction                                   │
│                                                           │
│  Database (SQLite/PostgreSQL)                             │
│  ├─ Users (students, teachers, admins)                   │
│  ├─ Videos (original content)                            │
│  ├─ Watermarked Videos (student-specific)                │
│  ├─ Access Grants (permissions)                          │
│  └─ Leak Reports (forensics)                             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Key Features

🔐 **Security**
- Dual watermarking (video + audio)
- Invisible, robust watermarks
- Forensic leak detection
- Automatic ban system

👨‍🏫 **For Teachers**
- Upload videos
- Manage student access
- Track views
- Detect leaks

👨‍🎓 **For Students**
- Watch watermarked videos
- Secure streaming
- No download option

👨‍💼 **For Admins**
- Full platform control
- User management
- Analytics dashboard
- Leak investigations

## API Endpoints (Quick Reference)

### Authentication
- `POST /api/auth/login` - Login
- `POST /api/auth/register` - Register
- `GET /api/auth/me` - Get current user

### Videos
- `GET /api/videos` - List videos
- `POST /api/videos/upload` - Upload video
- `GET /api/videos/{id}/stream` - Stream video

### Access Control
- `POST /api/videos/{id}/access` - Grant access
- `DELETE /api/videos/{id}/access/{student_id}` - Revoke

### Leak Detection
- `POST /api/leaks/detect` - Analyze video
- `GET /api/leaks` - List reports
- `PUT /api/leaks/{id}` - Update report

## Common Commands

### Reset Database
```bash
cd backend
rm vvm_school.db
python
from app import app, db
with app.app_context():
    db.create_all()
exit()
```

### Clear Uploads
```bash
cd backend
rm -rf uploads/* processed_videos/*
```

### Rebuild Frontend
```bash
cd frontend
npm run build
```

### Check Logs
```bash
# Backend logs in terminal
# Frontend logs in browser console (F12)
```

## Troubleshooting

### Backend won't start
- Ensure virtual environment is activated
- Check if port 5000 is available
- Verify all dependencies installed

### Frontend won't start
- Delete `node_modules` and reinstall
- Clear npm cache: `npm cache clean --force`
- Check if port 3000 is available

### Video upload fails
- Check file size (default max: 2GB)
- Verify FFmpeg is installed: `ffmpeg -version`
- Check upload folder permissions

### Watermark detection fails
- Ensure video hasn't been heavily re-encoded
- Check both video and audio streams
- Video may be too corrupted

## Configuration

### Change Upload Size Limit
Edit `backend/config.py`:
```python
MAX_CONTENT_LENGTH = 5 * 1024 * 1024 * 1024  # 5GB
```

### Change Watermark Strength
Edit `backend/watermark/embedder.py`:
```python
VIDEO_STRENGTH = 50  # Increase for stronger watermark
```

### Change Port
```bash
# Backend
python app.py --port 5001

# Frontend
PORT=3001 npm start
```

## Production Deployment

For production, you need:

1. **HTTPS** (SSL certificate)
2. **Production database** (PostgreSQL)
3. **Reverse proxy** (Nginx)
4. **Process manager** (systemd/supervisor)
5. **Strong secrets** (environment variables)

See `INSTALLATION.md` for full production setup.

## Getting Help

1. Check error messages
2. Review documentation
3. Ensure all prerequisites met
4. Verify file permissions
5. Check FFmpeg installation

## Next Steps

- [ ] Change admin password
- [ ] Create user accounts
- [ ] Upload test videos
- [ ] Test watermarking
- [ ] Test leak detection
- [ ] Configure for your needs
- [ ] Deploy to production

---

**You're ready to go! Happy teaching! 📚🎓**
