"""
VVM Online School Platform - Main Application
Flask backend with video watermarking capabilities
"""

import os
import uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, get_jwt
)
from werkzeug.utils import secure_filename

from config import Config
from models import (
    db, User, Video, WatermarkedVideo, VideoAccess, 
    VideoViewLog, LeakReport, UserRole, init_db
)
from watermark import process_dual_watermark, extract_watermark, VideoWatermarkError


app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Initialize extensions
CORS(app)
jwt = JWTManager(app)
init_db(app)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_VIDEO_EXTENSIONS


def role_required(required_role):
    """Decorator to check user role"""
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            try:
                current_user_id = get_jwt_identity()
                app.logger.info(f"Role check for user ID: {current_user_id}, required role: {required_role}")
                
                user = User.query.get(current_user_id)
                
                if not user or not user.is_active:
                    app.logger.warning(f"User {current_user_id} not found or inactive")
                    return jsonify({'error': 'User not found or inactive'}), 403
                
                if user.is_banned:
                    app.logger.warning(f"User {current_user_id} is banned")
                    return jsonify({'error': 'User is banned', 'reason': user.ban_reason}), 403
                
                if isinstance(required_role, list):
                    if user.role not in [UserRole[r.upper()] for r in required_role]:
                        app.logger.warning(f"User {current_user_id} has role {user.role}, required: {required_role}")
                        return jsonify({'error': 'Insufficient permissions'}), 403
                else:
                    if user.role != UserRole[required_role.upper()]:
                        app.logger.warning(f"User {current_user_id} has role {user.role}, required: {required_role}")
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                app.logger.info(f"Role check passed for user {current_user_id} ({user.username})")
                return fn(*args, **kwargs)
            except Exception as e:
                app.logger.error(f"Error in role_required decorator: {str(e)}")
                return jsonify({'error': 'Authorization error', 'details': str(e)}), 401
        return wrapper
    return decorator


def get_current_user():
    """Get current authenticated user"""
    user_id = get_jwt_identity()
    return User.query.get(user_id)


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password', 'full_name']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    # Determine role (default to student, only admins can create teachers/admins)
    role = UserRole.STUDENT
    if 'role' in data and data['role'] in ['teacher', 'admin']:
        # This would need admin authentication in production
        role = UserRole[data['role'].upper()]
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        full_name=data['full_name'],
        role=role
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict()
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user and return JWT token"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Missing username or password'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is inactive'}), 403
    
    if user.is_banned:
        return jsonify({'error': 'Account is banned', 'reason': user.ban_reason}), 403
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': user.to_dict()
    }), 200


@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user_info():
    """Get current user information"""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict()}), 200


# ============================================================================
# VIDEO MANAGEMENT ROUTES (TEACHER)
# ============================================================================

@app.route('/api/videos', methods=['GET'])
@jwt_required()
def list_videos():
    """List all videos (filtered by role)"""
    user = get_current_user()
    
    if user.role == UserRole.TEACHER:
        # Teachers see only their uploaded videos
        videos = Video.query.filter_by(teacher_id=user.id).all()
    elif user.role == UserRole.STUDENT:
        # Students see only videos they have access to
        access_grants = VideoAccess.query.filter_by(
            student_id=user.id, 
            is_active=True
        ).all()
        video_ids = [grant.video_id for grant in access_grants]
        videos = Video.query.filter(Video.id.in_(video_ids), Video.is_active==True).all()
    else:  # Admin
        videos = Video.query.all()
    
    return jsonify({
        'videos': [video.to_dict(include_stats=True) for video in videos]
    }), 200


@app.route('/api/videos/<int:video_id>', methods=['GET'])
@jwt_required()
def get_video(video_id):
    """Get specific video details"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    
    # Check access permissions
    if user.role == UserRole.STUDENT:
        access = VideoAccess.query.filter_by(
            video_id=video_id,
            student_id=user.id,
            is_active=True
        ).first()
        if not access:
            return jsonify({'error': 'Access denied'}), 403
    elif user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    return jsonify({'video': video.to_dict(include_stats=True)}), 200


@app.route('/api/videos/upload', methods=['POST'])
@role_required(['teacher', 'admin'])
def upload_video():
    """Upload a new video (teachers only)"""
    app.logger.info(f"Video upload request from user: {get_jwt_identity()}")
    app.logger.info(f"Content-Type: {request.content_type}")
    app.logger.info(f"Files: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Get metadata
    title = request.form.get('title', file.filename)
    description = request.form.get('description', '')
    course_name = request.form.get('course_name', '')
    subject = request.form.get('subject', '')
    
    # Save file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    file.save(file_path)
    
    file_size = os.path.getsize(file_path)
    
    # Create video record
    user = get_current_user()
    video = Video(
        title=title,
        description=description,
        original_filename=filename,
        file_path=file_path,
        file_size=file_size,
        teacher_id=user.id,
        course_name=course_name,
        subject=subject,
        processing_status='ready'
    )
    
    # Get video properties (optional, can be done async)
    try:
        from watermark.embedder import get_video_properties
        duration, fps, width, height = get_video_properties(file_path)
        video.duration = duration
        video.resolution = f"{width}x{height}"
    except Exception as e:
        app.logger.warning(f"Could not extract video properties: {e}")
    
    db.session.add(video)
    db.session.commit()
    
    return jsonify({
        'message': 'Video uploaded successfully',
        'video': video.to_dict()
    }), 201


@app.route('/api/videos/<int:video_id>', methods=['PUT'])
@role_required(['teacher', 'admin'])
def update_video(video_id):
    """Update video metadata"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    if user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    
    if 'title' in data:
        video.title = data['title']
    if 'description' in data:
        video.description = data['description']
    if 'course_name' in data:
        video.course_name = data['course_name']
    if 'subject' in data:
        video.subject = data['subject']
    if 'is_active' in data:
        video.is_active = data['is_active']
    
    db.session.commit()
    
    return jsonify({
        'message': 'Video updated successfully',
        'video': video.to_dict()
    }), 200


@app.route('/api/videos/<int:video_id>', methods=['DELETE'])
@role_required(['teacher', 'admin'])
def delete_video(video_id):
    """Delete a video"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    if user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Delete physical files
    try:
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
        
        # Delete watermarked versions
        for wm_video in video.watermarked_versions:
            if os.path.exists(wm_video.file_path):
                os.remove(wm_video.file_path)
    except Exception as e:
        app.logger.error(f"Error deleting files: {e}")
    
    db.session.delete(video)
    db.session.commit()
    
    return jsonify({'message': 'Video deleted successfully'}), 200


# ============================================================================
# VIDEO ACCESS MANAGEMENT
# ============================================================================

@app.route('/api/videos/<int:video_id>/access', methods=['GET'])
@role_required(['teacher', 'admin'])
def get_video_access(video_id):
    """Get list of students with access to video"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    if user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    access_grants = VideoAccess.query.filter_by(video_id=video_id).all()
    
    return jsonify({
        'access_grants': [grant.to_dict() for grant in access_grants]
    }), 200


@app.route('/api/videos/<int:video_id>/access', methods=['POST'])
@role_required(['teacher', 'admin'])
def grant_video_access(video_id):
    """Grant access to students"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    if user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    student_ids = data.get('student_ids', [])
    
    if not student_ids:
        return jsonify({'error': 'No student IDs provided'}), 400
    
    granted = []
    for student_id in student_ids:
        student = User.query.filter_by(id=student_id, role=UserRole.STUDENT).first()
        if not student:
            continue
        
        # Check if access already exists
        existing = VideoAccess.query.filter_by(
            video_id=video_id,
            student_id=student_id
        ).first()
        
        if existing:
            existing.is_active = True
            granted.append(existing)
        else:
            access = VideoAccess(
                video_id=video_id,
                student_id=student_id,
                granted_by=user.id
            )
            db.session.add(access)
            granted.append(access)
    
    db.session.commit()
    
    return jsonify({
        'message': f'Access granted to {len(granted)} students',
        'access_grants': [g.to_dict() for g in granted]
    }), 201


@app.route('/api/videos/<int:video_id>/access/<int:student_id>', methods=['DELETE'])
@role_required(['teacher', 'admin'])
def revoke_video_access(video_id, student_id):
    """Revoke student access to video"""
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    user = get_current_user()
    if user.role == UserRole.TEACHER and video.teacher_id != user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    access = VideoAccess.query.filter_by(
        video_id=video_id,
        student_id=student_id
    ).first()
    
    if not access:
        return jsonify({'error': 'Access grant not found'}), 404
    
    access.is_active = False
    db.session.commit()
    
    return jsonify({'message': 'Access revoked successfully'}), 200


# ============================================================================
# STUDENT VIDEO STREAMING
# ============================================================================

@app.route('/api/videos/<int:video_id>/stream', methods=['GET'])
@jwt_required()
def stream_video(video_id):
    """Stream watermarked video to student"""
    user = get_current_user()
    
    # Check if student has access
    if user.role == UserRole.STUDENT:
        access = VideoAccess.query.filter_by(
            video_id=video_id,
            student_id=user.id,
            is_active=True
        ).first()
        
        if not access:
            return jsonify({'error': 'Access denied'}), 403
    
    # Get or create watermarked version
    watermarked = WatermarkedVideo.query.filter_by(
        video_id=video_id,
        student_id=user.id
    ).first()
    
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    
    if not watermarked:
        # Generate watermarked version
        try:
            unique_filename = f"wm_{user.id}_{video_id}_{uuid.uuid4()}.mp4"
            output_path = os.path.join(Config.PROCESSED_FOLDER, unique_filename)
            
            result = process_dual_watermark(
                video.file_path,
                output_path,
                str(user.id),
                str(video_id)
            )
            
            watermarked = WatermarkedVideo(
                video_id=video_id,
                student_id=user.id,
                watermark_id=result['watermark_id'],
                file_path=output_path,
                file_size=os.path.getsize(output_path)
            )
            
            db.session.add(watermarked)
            db.session.commit()
            
        except VideoWatermarkError as e:
            app.logger.error(f"Watermarking failed: {e}")
            return jsonify({'error': 'Failed to process video'}), 500
    
    # Update access stats
    watermarked.last_accessed = datetime.utcnow()
    watermarked.access_count += 1
    
    # Log view
    view_log = VideoViewLog(
        user_id=user.id,
        video_id=video_id,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )
    db.session.add(view_log)
    db.session.commit()
    
    # Return file
    return send_file(
        watermarked.file_path,
        mimetype='video/mp4',
        as_attachment=False
    )


# ============================================================================
# LEAK DETECTION AND REPORTING
# ============================================================================

@app.route('/api/leaks/detect', methods=['POST'])
@role_required(['teacher', 'admin'])
def detect_leak():
    """Detect leak by analyzing uploaded video sample"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save temporary file
    temp_filename = f"temp_{uuid.uuid4()}_{secure_filename(file.filename)}"
    temp_path = os.path.join(Config.UPLOAD_FOLDER, temp_filename)
    file.save(temp_path)
    
    try:
        # Extract watermark
        result = extract_watermark(temp_path)
        
        if not result.get('match'):
            return jsonify({
                'error': 'Could not extract watermark',
                'details': result
            }), 400
        
        watermark_id = result['final_id']
        
        # Find watermarked video
        watermarked = WatermarkedVideo.query.filter_by(
            watermark_id=str(watermark_id)
        ).first()
        
        if not watermarked:
            return jsonify({
                'error': 'Watermark ID not found in database',
                'watermark_id': watermark_id
            }), 404
        
        # Create leak report
        user = get_current_user()
        leak_report = LeakReport(
            video_id=watermarked.video_id,
            suspected_user_id=watermarked.student_id,
            watermark_id=str(watermark_id),
            detection_method='manual',
            reported_by=user.id,
            status='pending',
            evidence_file_path=temp_path
        )
        
        db.session.add(leak_report)
        db.session.commit()
        
        return jsonify({
            'message': 'Leak detected successfully',
            'leak_report': leak_report.to_dict(),
            'suspected_user': watermarked.student.to_dict(),
            'video': watermarked.original_video.to_dict(),
            'extraction_details': result
        }), 200
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        app.logger.error(f"Leak detection failed: {e}")
        return jsonify({'error': 'Detection failed', 'details': str(e)}), 500


@app.route('/api/leaks', methods=['GET'])
@role_required(['teacher', 'admin'])
def list_leak_reports():
    """List all leak reports"""
    user = get_current_user()
    
    if user.role == UserRole.TEACHER:
        # Teachers see leaks for their videos only
        video_ids = [v.id for v in Video.query.filter_by(teacher_id=user.id).all()]
        reports = LeakReport.query.filter(LeakReport.video_id.in_(video_ids)).all()
    else:  # Admin
        reports = LeakReport.query.all()
    
    return jsonify({
        'leak_reports': [report.to_dict() for report in reports]
    }), 200


@app.route('/api/leaks/<int:report_id>', methods=['PUT'])
@role_required(['teacher', 'admin'])
def update_leak_report(report_id):
    """Update leak report status"""
    report = LeakReport.query.get(report_id)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    data = request.get_json()
    
    if 'status' in data:
        report.status = data['status']
    if 'notes' in data:
        report.notes = data['notes']
    
    # If confirmed, ban the user
    if data.get('status') == 'confirmed' and data.get('ban_user'):
        suspected_user = User.query.get(report.suspected_user_id)
        if suspected_user:
            suspected_user.is_banned = True
            suspected_user.ban_reason = f"Video leak confirmed - Report #{report.id}"
    
    db.session.commit()
    
    return jsonify({
        'message': 'Report updated successfully',
        'leak_report': report.to_dict()
    }), 200


# ============================================================================
# USER MANAGEMENT (ADMIN)
# ============================================================================

@app.route('/api/users', methods=['GET'])
@role_required('admin')
def list_users():
    """List all users (admin only)"""
    role_filter = request.args.get('role')
    
    query = User.query
    if role_filter:
        query = query.filter_by(role=UserRole[role_filter.upper()])
    
    users = query.all()
    
    return jsonify({
        'users': [user.to_dict(include_sensitive=True) for user in users]
    }), 200


@app.route('/api/users/<int:user_id>/ban', methods=['POST'])
@role_required('admin')
def ban_user(user_id):
    """Ban a user"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    reason = data.get('reason', 'No reason provided')
    
    user.is_banned = True
    user.ban_reason = reason
    db.session.commit()
    
    return jsonify({
        'message': 'User banned successfully',
        'user': user.to_dict(include_sensitive=True)
    }), 200


@app.route('/api/users/<int:user_id>/unban', methods=['POST'])
@role_required('admin')
def unban_user(user_id):
    """Unban a user"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    user.is_banned = False
    user.ban_reason = None
    db.session.commit()
    
    return jsonify({
        'message': 'User unbanned successfully',
        'user': user.to_dict()
    }), 200


# ============================================================================
# STATISTICS AND ANALYTICS
# ============================================================================

@app.route('/api/stats', methods=['GET'])
@jwt_required()
def get_statistics():
    """Get platform statistics"""
    user = get_current_user()
    
    stats = {}
    
    if user.role == UserRole.ADMIN:
        stats = {
            'total_users': User.query.count(),
            'total_students': User.query.filter_by(role=UserRole.STUDENT).count(),
            'total_teachers': User.query.filter_by(role=UserRole.TEACHER).count(),
            'total_videos': Video.query.count(),
            'total_watermarked': WatermarkedVideo.query.count(),
            'total_leaks': LeakReport.query.count(),
            'pending_leaks': LeakReport.query.filter_by(status='pending').count(),
            'confirmed_leaks': LeakReport.query.filter_by(status='confirmed').count(),
            'banned_users': User.query.filter_by(is_banned=True).count()
        }
    elif user.role == UserRole.TEACHER:
        stats = {
            'uploaded_videos': Video.query.filter_by(teacher_id=user.id).count(),
            'total_students': db.session.query(VideoAccess.student_id).filter(
                VideoAccess.video_id.in_([v.id for v in Video.query.filter_by(teacher_id=user.id)])
            ).distinct().count(),
            'total_views': db.session.query(VideoViewLog).filter(
                VideoViewLog.video_id.in_([v.id for v in Video.query.filter_by(teacher_id=user.id)])
            ).count()
        }
    else:  # Student
        stats = {
            'accessible_videos': VideoAccess.query.filter_by(
                student_id=user.id,
                is_active=True
            ).count(),
            'watched_videos': VideoViewLog.query.filter_by(user_id=user.id).count()
        }
    
    return jsonify({'stats': stats}), 200


# ============================================================================
# JWT ERROR HANDLERS
# ============================================================================

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    app.logger.warning(f"Expired token from user: {jwt_payload.get('sub')}")
    return jsonify({
        'error': 'Token has expired',
        'msg': 'The token has expired. Please log in again.'
    }), 401


@jwt.invalid_token_loader
def invalid_token_callback(error):
    app.logger.warning(f"Invalid token: {error}")
    return jsonify({
        'error': 'Invalid token',
        'msg': 'Signature verification failed. Please log in again.'
    }), 422


@jwt.unauthorized_loader
def missing_token_callback(error):
    app.logger.warning(f"Missing token: {error}")
    return jsonify({
        'error': 'Missing authorization',
        'msg': 'Authorization header is missing.'
    }), 401


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/')
def index():
    return jsonify({
        'name': 'VVM Online School Platform',
        'version': '1.0.0',
        'description': 'Secure video platform with watermarking'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
