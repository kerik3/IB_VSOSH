# VVM Platform - Detailed Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 10GB free space (more for video storage)
- **CPU**: Dual-core processor (Quad-core recommended)

### Software Requirements
- Python 3.8 or higher
- Node.js 16 or higher
- FFmpeg 4.0 or higher
- pip (Python package manager)
- npm (Node package manager)

## Step-by-Step Installation

### 1. Install System Dependencies

#### Windows

**Install Python:**
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Verify: `python --version`

**Install Node.js:**
1. Download from https://nodejs.org/
2. Run installer (LTS version recommended)
3. Verify: `node --version` and `npm --version`

**Install FFmpeg:**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH
4. Verify: `ffmpeg -version`

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python
sudo apt install python3 python3-pip python3-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install FFmpeg
sudo apt install ffmpeg

# Verify installations
python3 --version
node --version
npm --version
ffmpeg -version
```

#### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python node ffmpeg

# Verify installations
python3 --version
node --version
npm --version
ffmpeg -version
```

### 2. Clone or Download Project

```bash
# If using git
git clone <repository-url>
cd IB_VSOSH

# Or download and extract ZIP file
```

### 3. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Create directories:**
```bash
mkdir uploads
mkdir processed_videos
```

**Initialize database:**
```bash
python
```

```python
from app import app, db
with app.app_context():
    db.create_all()
    print("Database initialized successfully!")
exit()
```

### 4. Frontend Setup

```bash
# Navigate to frontend (from project root)
cd frontend

# Install dependencies
npm install

# This may take a few minutes
```

### 5. Configuration

**Backend Configuration:**

The backend will use default SQLite database. For production, you can configure:

Create `backend/.env`:
```env
SECRET_KEY=your-very-secure-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
DATABASE_URL=sqlite:///vvm_school.db
```

**Frontend Configuration:**

Create `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:5000/api
```

### 6. Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
# Activate venv if not already active
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

The application will open automatically in your browser at `http://localhost:3000`

### 7. First Login

**Default admin credentials:**
- Username: `admin`
- Password: `admin123`

⚠️ **IMPORTANT**: Change the admin password immediately after first login!

### 8. Create Test Accounts

**Create a Teacher Account:**
1. Go to Register page
2. Fill in details
3. Select "Teacher" role
4. Register

**Create a Student Account:**
1. Go to Register page
2. Fill in details
3. Select "Student" role
4. Register

## Verification Checklist

- [ ] Python is installed and in PATH
- [ ] Node.js is installed and in PATH
- [ ] FFmpeg is installed and in PATH
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Database initialized
- [ ] Backend server starts without errors
- [ ] Frontend builds and runs
- [ ] Can access login page
- [ ] Can login with admin credentials

## Troubleshooting Common Issues

### Issue: "pip not found"
**Solution:**
```bash
# Windows
python -m pip install --upgrade pip

# Linux/Mac
python3 -m pip install --upgrade pip
```

### Issue: "FFmpeg not found"
**Solution:**
- Ensure FFmpeg is in your system PATH
- Test with: `ffmpeg -version`
- Restart terminal after adding to PATH

### Issue: "Port 5000 already in use"
**Solution:**
```bash
# Find process using port 5000
# Windows
netstat -ano | findstr :5000

# Linux/Mac
lsof -i :5000

# Kill the process or change port in app.py
```

### Issue: "Node modules not found"
**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue: "Database locked"
**Solution:**
```bash
# Stop all running processes
# Delete database file
rm backend/vvm_school.db

# Reinitialize
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.create_all()
>>> exit()
```

### Issue: "Module not found" errors
**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## Testing the Installation

### Test 1: Upload a Video (Teacher)

1. Login as teacher
2. Go to "Videos" → "Upload Video"
3. Upload a small test video (recommend < 100MB for testing)
4. Check console for errors

### Test 2: Grant Access (Teacher)

1. Create a student account
2. As teacher, go to video → "Manage Access"
3. Select the student
4. Grant access

### Test 3: Watch Video (Student)

1. Login as student
2. Go to "Videos"
3. Click "Watch Video"
4. Video should stream with watermark

### Test 4: Detect Leak (Teacher/Admin)

1. Download the watermarked video
2. Go to "Leak Detection"
3. Upload the video
4. System should identify the student

## Production Deployment

For production deployment, additional steps are required:

1. **Use a production database** (PostgreSQL/MySQL)
2. **Set strong SECRET_KEY values**
3. **Use a production WSGI server** (Gunicorn)
4. **Set up reverse proxy** (Nginx)
5. **Enable HTTPS**
6. **Set up proper backup system**
7. **Configure firewall rules**
8. **Set up monitoring and logging**

See `DEPLOYMENT.md` for detailed production setup instructions.

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Check that virtual environment is activated
5. Verify FFmpeg is working: `ffmpeg -version`
6. Check file permissions in upload directories

## Next Steps

After successful installation:

1. Change admin password
2. Create teacher accounts
3. Create student accounts
4. Upload test videos
5. Test the watermarking system
6. Test leak detection
7. Explore admin panel features

---

**Congratulations! Your VVM Platform is ready to use! 🎉**
