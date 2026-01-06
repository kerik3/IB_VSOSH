import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { videoAPI } from '../services/api';
import { toast } from 'react-toastify';
import { FiVideo, FiTrash2, FiEdit, FiUsers, FiPlay, FiUpload } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const VideoCard = ({ video, onDelete, onEdit, onManageAccess, onView, isTeacher }) => {
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' Б';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' КБ';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' МБ';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' ГБ';
  };

  const formatDuration = (seconds) => {
    if (!seconds) return 'Н/Д';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusLabel = (status) => {
    switch(status) {
      case 'ready': return 'Готово';
      case 'pending': return 'Ожидание';
      case 'processing': return 'Обработка';
      case 'failed': return 'Ошибка';
      default: return status;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all p-6 border border-gray-100">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-primary-100 rounded-lg">
            <FiVideo className="w-6 h-6 text-primary-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{video.title}</h3>
            <p className="text-sm text-gray-500">{video.course_name || 'Без курса'}</p>
          </div>
        </div>
        {!video.is_active && (
          <span className="px-2 py-1 text-xs font-semibold text-red-600 bg-red-100 rounded-full">
            Неактивно
          </span>
        )}
      </div>

      <p className="text-gray-600 text-sm mb-4 line-clamp-2">
        {video.description || 'Без описания'}
      </p>

      <div className="grid grid-cols-2 gap-3 mb-4 text-sm">
        <div>
          <span className="text-gray-500">Размер:</span>
          <span className="font-semibold ml-2">{formatFileSize(video.file_size)}</span>
        </div>
        <div>
          <span className="text-gray-500">Длительность:</span>
          <span className="font-semibold ml-2">{formatDuration(video.duration)}</span>
        </div>
        <div>
          <span className="text-gray-500">Разрешение:</span>
          <span className="font-semibold ml-2">{video.resolution || 'Н/Д'}</span>
        </div>
        <div>
          <span className="text-gray-500">Статус:</span>
          <span className="font-semibold ml-2">{getStatusLabel(video.processing_status)}</span>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 pt-4 border-t border-gray-200">
        {isTeacher ? (
          <>
            <button
              onClick={() => onManageAccess(video.id)}
              className="flex items-center space-x-1 px-3 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm"
            >
              <FiUsers className="w-4 h-4" />
              <span>Доступ</span>
            </button>
            <button
              onClick={() => onEdit(video.id)}
              className="flex items-center space-x-1 px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm"
            >
              <FiEdit className="w-4 h-4" />
              <span>Изменить</span>
            </button>
            <button
              onClick={() => onDelete(video.id)}
              className="flex items-center space-x-1 px-3 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors text-sm"
            >
              <FiTrash2 className="w-4 h-4" />
              <span>Удалить</span>
            </button>
          </>
        ) : (
          <button
            onClick={() => onView(video.id)}
            className="flex items-center space-x-1 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors text-sm"
          >
            <FiPlay className="w-4 h-4" />
            <span>Смотреть</span>
          </button>
        )}
      </div>
    </div>
  );
};

const VideoList = () => {
  const { isTeacher, isAdmin } = useAuth();
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await videoAPI.list();
      setVideos(response.data.videos);
    } catch (error) {
      toast.error('Ошибка загрузки видео');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Вы уверены, что хотите удалить это видео?')) {
      return;
    }

    try {
      await videoAPI.delete(id);
      toast.success('Видео успешно удалено');
      fetchVideos();
    } catch (error) {
      toast.error('Ошибка удаления видео');
    }
  };

  const handleEdit = (id) => {
    navigate(`/videos/${id}/edit`);
  };

  const handleManageAccess = (id) => {
    navigate(`/videos/${id}/access`);
  };

  const handleView = (id) => {
    navigate(`/videos/${id}/view`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Видео</h1>
          <p className="text-gray-600 mt-2">
            {isTeacher || isAdmin ? 'Управление видеотекой' : 'Доступные вам видео'}
          </p>
        </div>
        {(isTeacher || isAdmin) && (
          <button
            onClick={() => navigate('/videos/upload')}
            className="flex items-center space-x-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors shadow-lg"
          >
            <FiUpload className="w-5 h-5" />
            <span>Загрузить видео</span>
          </button>
        )}
      </div>

      {videos.length === 0 ? (
        <div className="text-center py-20">
          <FiVideo className="w-20 h-20 mx-auto text-gray-300 mb-4" />
          <h3 className="text-xl font-semibold text-gray-600 mb-2">Видео пока нет</h3>
          <p className="text-gray-500">
            {isTeacher || isAdmin
              ? 'Загрузите первое видео, чтобы начать'
              : 'Вам пока не предоставлен доступ к видео'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {videos.map((video) => (
            <VideoCard
              key={video.id}
              video={video}
              onDelete={handleDelete}
              onEdit={handleEdit}
              onManageAccess={handleManageAccess}
              onView={handleView}
              isTeacher={isTeacher || isAdmin}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default VideoList;
