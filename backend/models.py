"""
Database models for VVM Online School Platform
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from enum import Enum

db = SQLAlchemy()
bcrypt = Bcrypt()


class UserRole(Enum):
    """User role enumeration"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"


class User(db.Model):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.Enum(UserRole), nullable=False, default=UserRole.STUDENT)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_banned = db.Column(db.Boolean, default=False, nullable=False)
    ban_reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    uploaded_videos = db.relationship('Video', back_populates='teacher', lazy='dynamic',
                                     foreign_keys='Video.teacher_id')
    watermarked_videos = db.relationship('WatermarkedVideo', back_populates='student', 
                                        lazy='dynamic', cascade='all, delete-orphan')
    view_logs = db.relationship('VideoViewLog', back_populates='user', lazy='dynamic',
                               cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        """Check if password matches hash"""
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role.value,
            'is_active': self.is_active,
            'is_banned': self.is_banned,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        
        if include_sensitive and self.is_banned:
            data['ban_reason'] = self.ban_reason
        
        return data
    
    def __repr__(self):
        return f'<User {self.username} ({self.role.value})>'


class Video(db.Model):
    """Original video uploaded by teacher"""
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)  # bytes
    duration = db.Column(db.Float, nullable=True)  # seconds
    resolution = db.Column(db.String(20), nullable=True)  # e.g., "1920x1080"
    
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_name = db.Column(db.String(200), nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    processing_status = db.Column(db.String(50), default='pending')  # pending, processing, ready, failed
    
    # Relationships
    teacher = db.relationship('User', back_populates='uploaded_videos', 
                             foreign_keys=[teacher_id])
    watermarked_versions = db.relationship('WatermarkedVideo', back_populates='original_video',
                                          lazy='dynamic', cascade='all, delete-orphan')
    access_grants = db.relationship('VideoAccess', back_populates='video',
                                   lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self, include_stats=False):
        """Convert video to dictionary"""
        data = {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'duration': self.duration,
            'resolution': self.resolution,
            'teacher_id': self.teacher_id,
            'teacher_name': self.teacher.full_name if self.teacher else None,
            'course_name': self.course_name,
            'subject': self.subject,
            'is_active': self.is_active,
            'uploaded_at': self.uploaded_at.isoformat(),
            'processing_status': self.processing_status
        }
        
        if include_stats:
            data['watermarked_count'] = self.watermarked_versions.count()
            data['access_grants_count'] = self.access_grants.filter_by(is_active=True).count()
        
        return data
    
    def __repr__(self):
        return f'<Video {self.id}: {self.title}>'


class WatermarkedVideo(db.Model):
    """Student-specific watermarked version of a video"""
    __tablename__ = 'watermarked_videos'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    watermark_id = db.Column(db.String(100), nullable=False, unique=True, index=True)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.BigInteger, nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_accessed = db.Column(db.DateTime, nullable=True)
    access_count = db.Column(db.Integer, default=0, nullable=False)
    
    # Relationships
    original_video = db.relationship('Video', back_populates='watermarked_versions')
    student = db.relationship('User', back_populates='watermarked_videos')
    
    def to_dict(self):
        """Convert watermarked video to dictionary"""
        return {
            'id': self.id,
            'video_id': self.video_id,
            'student_id': self.student_id,
            'student_name': self.student.full_name if self.student else None,
            'watermark_id': self.watermark_id,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_count': self.access_count
        }
    
    def __repr__(self):
        return f'<WatermarkedVideo {self.id}: Video{self.video_id} for User{self.student_id}>'


class VideoAccess(db.Model):
    """Access control for videos - which students can access which videos"""
    __tablename__ = 'video_access'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    granted_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    granted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    video = db.relationship('Video', back_populates='access_grants')
    student = db.relationship('User', foreign_keys=[student_id])
    granter = db.relationship('User', foreign_keys=[granted_by])
    
    # Unique constraint: one access grant per student per video
    __table_args__ = (
        db.UniqueConstraint('video_id', 'student_id', name='uq_video_student_access'),
    )
    
    def to_dict(self):
        """Convert access grant to dictionary"""
        return {
            'id': self.id,
            'video_id': self.video_id,
            'student_id': self.student_id,
            'student_name': self.student.full_name if self.student else None,
            'granted_by': self.granted_by,
            'granted_at': self.granted_at.isoformat(),
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    def __repr__(self):
        return f'<VideoAccess: Video{self.video_id} -> User{self.student_id}>'


class VideoViewLog(db.Model):
    """Log of video views for analytics and leak detection"""
    __tablename__ = 'video_view_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    
    viewed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)  # IPv6 compatible
    user_agent = db.Column(db.String(500), nullable=True)
    watch_duration = db.Column(db.Integer, nullable=True)  # seconds
    
    # Relationships
    user = db.relationship('User', back_populates='view_logs')
    video = db.relationship('Video')
    
    def to_dict(self):
        """Convert view log to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.full_name if self.user else None,
            'video_id': self.video_id,
            'video_title': self.video.title if self.video else None,
            'viewed_at': self.viewed_at.isoformat(),
            'ip_address': self.ip_address,
            'watch_duration': self.watch_duration
        }
    
    def __repr__(self):
        return f'<ViewLog: User{self.user_id} watched Video{self.video_id}>'


class LeakReport(db.Model):
    """Report of detected video leaks"""
    __tablename__ = 'leak_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('videos.id'), nullable=False)
    suspected_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    watermark_id = db.Column(db.String(100), nullable=False, index=True)
    detection_method = db.Column(db.String(50), nullable=True)  # manual, automated
    
    reported_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reported_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    status = db.Column(db.String(50), default='pending')  # pending, investigating, confirmed, false_positive
    notes = db.Column(db.Text, nullable=True)
    
    # Evidence file (leaked video sample for verification)
    evidence_file_path = db.Column(db.String(500), nullable=True)
    
    # Relationships
    video = db.relationship('Video')
    suspected_user = db.relationship('User', foreign_keys=[suspected_user_id])
    reporter = db.relationship('User', foreign_keys=[reported_by])
    
    def to_dict(self):
        """Convert leak report to dictionary"""
        return {
            'id': self.id,
            'video_id': self.video_id,
            'video_title': self.video.title if self.video else None,
            'suspected_user_id': self.suspected_user_id,
            'suspected_user_name': self.suspected_user.full_name if self.suspected_user else None,
            'watermark_id': self.watermark_id,
            'detection_method': self.detection_method,
            'reported_by': self.reported_by,
            'reporter_name': self.reporter.full_name if self.reporter else None,
            'reported_at': self.reported_at.isoformat(),
            'status': self.status,
            'notes': self.notes
        }
    
    def __repr__(self):
        return f'<LeakReport {self.id}: User{self.suspected_user_id} - {self.status}>'


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    bcrypt.init_app(app)
    
    with app.app_context():
        db.create_all()
        
        # Create default admin if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@vvm.school',
                full_name='System Administrator',
                role=UserRole.ADMIN,
                is_active=True
            )
            admin.set_password('admin123')  # Change in production!
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created (username: admin, password: admin123)")
