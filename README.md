# VVM Online School Platform

A secure online learning platform with advanced video watermarking technology to prevent unauthorized content sharing.

## 🔐 Key Features

### Security
- **Dual Watermarking System**: Embeds unique watermarks in both video and audio streams
- **DCT-based Video Watermarking**: Invisible watermarks using Discrete Cosine Transform
- **FFT-based Audio Watermarking**: Frequency domain watermark embedding
- **Forensic Tracking**: Identify the source of leaked videos
- **Automatic Ban System**: Ban users who leak content

### For Teachers
- Upload and manage educational videos
- Grant/revoke student access to content
- Track video views and analytics
- Detect and report content leaks
- Monitor student engagement

### For Students
- Access personalized watermarked videos
- Secure video streaming
- No ability to download or share videos
- Clear terms and conditions

### For Administrators
- Complete platform management
- User administration (ban/unban)
- Leak detection and investigation
- Platform-wide analytics and statistics

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **SQLAlchemy** - ORM for database management
- **Flask-JWT-Extended** - Authentication
- **OpenCV** - Video processing
- **NumPy & SciPy** - Mathematical operations
- **FFmpeg** - Video encoding/decoding
- **Pydub** - Audio processing

### Frontend
- **React 18** - UI framework
- **React Router** - Navigation
- **TailwindCSS** - Styling
- **Axios** - HTTP client
- **React Toastify** - Notifications
- **React Dropzone** - File uploads

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- FFmpeg installed and in PATH
- Redis (optional, for background tasks)

### Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
```

3. **Activate virtual environment:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Create .env file:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Initialize database:**
```bash
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.create_all()
>>> exit()
```

7. **Run the server:**
```bash
python app.py
```

Backend will run on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Create .env file:**
```bash
cp .env.example .env
```

4. **Start development server:**
```bash
npm start
```

Frontend will run on `http://localhost:3000`

## 🚀 Quick Start

### Default Admin Account
After first run, a default admin account is created:
- **Username:** `admin`
- **Password:** `admin123`
- **⚠️ Change this password immediately in production!**

### Creating Users

1. **Register as Teacher:**
   - Go to `/register`
   - Select "Teacher" role
   - Fill in details

2. **Register as Student:**
   - Go to `/register`
   - Select "Student" role
   - Fill in details

### Uploading Videos (Teacher)

1. Login as teacher
2. Navigate to "Videos" → "Upload Video"
3. Drag & drop your video file
4. Fill in video metadata
5. Click "Upload Video"

### Granting Access (Teacher)

1. Go to "Videos"
2. Click "Manage Access" on a video
3. Select students to grant access
4. Click "Grant Access"

### Watching Videos (Student)

1. Login as student
2. Navigate to "Videos"
3. Click "Watch Video" on any available video
4. Video will be automatically watermarked with your unique ID

### Detecting Leaks (Teacher/Admin)

1. Navigate to "Leak Detection"
2. Upload the suspected leaked video
3. Click "Analyze Watermark"
4. System will identify the source user
5. Take appropriate action (confirm/ban)

## 📚 API Documentation

### Authentication Endpoints

#### POST `/api/auth/register`
Register a new user
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string",
  "role": "student|teacher"
}
```

#### POST `/api/auth/login`
Login and get JWT token
```json
{
  "username": "string",
  "password": "string"
}
```

#### GET `/api/auth/me`
Get current user info (requires authentication)

### Video Endpoints

#### GET `/api/videos`
List all videos (filtered by role)

#### POST `/api/videos/upload`
Upload new video (teachers only)

#### GET `/api/videos/{id}`
Get specific video details

#### PUT `/api/videos/{id}`
Update video metadata

#### DELETE `/api/videos/{id}`
Delete video

#### GET `/api/videos/{id}/stream`
Stream watermarked video (students)

### Access Control

#### GET `/api/videos/{id}/access`
Get access list for video

#### POST `/api/videos/{id}/access`
Grant access to students

#### DELETE `/api/videos/{id}/access/{student_id}`
Revoke student access

### Leak Detection

#### POST `/api/leaks/detect`
Analyze video for watermark

#### GET `/api/leaks`
List all leak reports

#### PUT `/api/leaks/{id}`
Update leak report status

### User Management (Admin)

#### GET `/api/users`
List all users

#### POST `/api/users/{id}/ban`
Ban a user

#### POST `/api/users/{id}/unban`
Unban a user

## 🔬 Watermarking Technology

### How It Works

1. **Video Watermarking (DCT Method)**
   - Converts video frames to YCrCb color space
   - Applies DCT to 8x8 pixel blocks
   - Modifies mid-frequency coefficients to embed user ID
   - Uses first and last second of video for redundancy

2. **Audio Watermarking (FFT Method)**
   - Converts audio to frequency domain using FFT
   - Modifies specific frequency bands
   - Embeds watermark in non-silent segments
   - Maintains audio quality while embedding data

3. **Extraction Process**
   - Analyzes both video and audio streams
   - Uses voting system for robust detection
   - Calculates confidence scores
   - Verifies consistency between streams

### Security Features

- **32-bit Watermark ID**: Over 4 billion unique identifiers
- **Secure ID Generation**: Combines user ID, video ID, and random salt
- **Dual Stream Verification**: Both video and audio must match
- **Invisible Watermarking**: No perceptible quality loss
- **Robust Against Compression**: Survives re-encoding and compression

## 🎨 UI Screenshots

### Login Page
Modern, secure authentication with role-based access

### Dashboard
Role-specific dashboards with statistics and quick actions

### Video Library
Grid view of videos with metadata and status indicators

### Video Player
Secure video player with watermark notice

### Leak Detection
Upload and analyze suspected leaked videos

### Admin Panel
Complete user and platform management

## 🔧 Configuration

### Backend Configuration (`backend/config.py`)

```python
SECRET_KEY = 'your-secret-key'
JWT_SECRET_KEY = 'your-jwt-secret'
DATABASE_URL = 'sqlite:///vvm_school.db'  # or PostgreSQL/MySQL
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
MAX_CONTENT_LENGTH = 2GB  # Max upload size
```

### Watermark Configuration (`backend/watermark/embedder.py`)

```python
SEGMENT_DURATION = 1  # Seconds to watermark at start/end
VIDEO_STRENGTH = 50  # DCT modification strength (higher = more robust)
AUDIO_CHUNK_SIZE = 4096  # FFT chunk size
ID_LENGTH = 32  # Watermark bit length
```

## 🐛 Troubleshooting

### FFmpeg Not Found
```bash
# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH

# Linux
sudo apt-get install ffmpeg

# Mac
brew install ffmpeg
```

### Port Already in Use
```bash
# Backend
python app.py --port 5001

# Frontend
PORT=3001 npm start
```

### Database Errors
```bash
# Reset database
rm backend/vvm_school.db
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.create_all()
```

## 📈 Performance Optimization

### For Large Videos
- Use background task queue (Celery + Redis)
- Implement chunked uploads
- Add progress notifications
- Cache watermarked videos

### For Multiple Users
- Use CDN for video delivery
- Implement video transcoding pipeline
- Add load balancing
- Use PostgreSQL instead of SQLite

## 🔒 Security Recommendations

1. **Change Default Credentials** immediately
2. **Use HTTPS** in production
3. **Set strong SECRET_KEY** values
4. **Enable CORS** only for trusted domains
5. **Implement rate limiting** on API endpoints
6. **Regular backups** of database
7. **Monitor leak reports** actively
8. **Use environment variables** for sensitive data

## 📝 License

This project is for educational and information security purposes.

## 👥 Support

For questions or issues:
- Check the troubleshooting section
- Review API documentation
- Contact system administrator

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Email notifications
- [ ] Advanced analytics dashboard
- [ ] Video transcoding for multiple qualities
- [ ] Mobile app
- [ ] Live streaming support
- [ ] Automated testing suite
- [ ] Docker containerization
- [ ] Cloud storage integration (AWS S3, Google Cloud)
- [ ] Machine learning for improved watermark detection

## 🙏 Acknowledgments

- OpenCV for video processing capabilities
- SciPy for mathematical operations
- React team for the excellent UI framework
- TailwindCSS for modern styling

---

**Built with ❤️ for secure online education**
