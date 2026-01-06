import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { videoAPI } from '../services/api';
import { toast } from 'react-toastify';
import { FiArrowLeft, FiInfo } from 'react-icons/fi';

const VideoPlayer = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [streamUrl, setStreamUrl] = useState(null);

  useEffect(() => {
    fetchVideo();
  }, [id]);

  const fetchVideo = async () => {
    try {
      const response = await videoAPI.get(id);
      setVideo(response.data.video);
      
      // Get stream URL with auth token
      const token = localStorage.getItem('token');
      const url = `${videoAPI.getStreamUrl(id)}?token=${token}`;
      setStreamUrl(url);
    } catch (error) {
      toast.error('Failed to load video');
      navigate('/videos');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!video) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Back Button */}
        <button
          onClick={() => navigate('/videos')}
          className="flex items-center space-x-2 text-white hover:text-primary-400 mb-6 transition-colors"
        >
          <FiArrowLeft className="w-5 h-5" />
          <span>Back to Videos</span>
        </button>

        {/* Video Player */}
        <div className="bg-black rounded-xl overflow-hidden shadow-2xl mb-6">
          <video
            controls
            className="w-full"
            style={{ maxHeight: '70vh' }}
            controlsList="nodownload"
            onContextMenu={(e) => e.preventDefault()}
          >
            <source src={streamUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>

        {/* Video Information */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">{video.title}</h1>
              <p className="text-gray-600">{video.course_name}</p>
            </div>
            <div className="flex items-center space-x-2 bg-primary-100 px-4 py-2 rounded-lg">
              <FiInfo className="text-primary-600" />
              <span className="text-sm font-semibold text-primary-700">
                This video is watermarked
              </span>
            </div>
          </div>

          {video.description && (
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
              <p className="text-gray-700 leading-relaxed">{video.description}</p>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
            <div>
              <p className="text-sm text-gray-500">Resolution</p>
              <p className="font-semibold text-gray-900">{video.resolution || 'N/A'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Duration</p>
              <p className="font-semibold text-gray-900">
                {video.duration ? `${Math.floor(video.duration / 60)}:${Math.floor(video.duration % 60).toString().padStart(2, '0')}` : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Teacher</p>
              <p className="font-semibold text-gray-900">{video.teacher_name}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Subject</p>
              <p className="font-semibold text-gray-900">{video.subject || 'N/A'}</p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800 flex items-start space-x-2">
              <FiInfo className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Important:</strong> This video contains a unique watermark tied to your account. 
                Unauthorized sharing or distribution is prohibited and will result in immediate action.
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
