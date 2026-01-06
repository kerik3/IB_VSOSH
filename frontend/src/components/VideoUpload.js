import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { videoAPI } from '../services/api';
import { toast } from 'react-toastify';
import { FiUploadCloud, FiVideo, FiX } from 'react-icons/fi';

const VideoUpload = () => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    course_name: '',
    subject: '',
  });
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
        if (!formData.title) {
          setFormData({ ...formData, title: acceptedFiles[0].name });
        }
      }
    },
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      toast.error('Выберите видеофайл');
      return;
    }

    setUploading(true);

    const data = new FormData();
    data.append('file', file);
    data.append('title', formData.title);
    data.append('description', formData.description);
    data.append('course_name', formData.course_name);
    data.append('subject', formData.subject);

    try {
      await videoAPI.upload(data, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(progress);
      });

      toast.success('Видео успешно загружено!');
      navigate('/videos');
    } catch (error) {
      toast.error('Ошибка загрузки видео');
      console.error('Upload error:', error);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' КБ';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' МБ';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' ГБ';
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Загрузка видео</h1>
        <p className="text-gray-600 mt-2">Загрузите новое видео для учеников</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File Upload Area */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Видеофайл *
          </label>
          
          {!file ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
              }`}
            >
              <input {...getInputProps()} />
              <FiUploadCloud className="w-16 h-16 mx-auto text-gray-400 mb-4" />
              <p className="text-lg font-semibold text-gray-700 mb-2">
                {isDragActive ? 'Отпустите файл здесь' : 'Перетащите видео сюда'}
              </p>
              <p className="text-sm text-gray-500">
                или нажмите для выбора (MP4, AVI, MOV, MKV, WebM)
              </p>
            </div>
          ) : (
            <div className="border-2 border-green-300 bg-green-50 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-green-500 rounded-lg">
                    <FiVideo className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-600">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setFile(null)}
                  className="p-2 text-red-500 hover:bg-red-100 rounded-lg transition-colors"
                >
                  <FiX className="w-6 h-6" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Video Metadata */}
        <div className="bg-white rounded-xl shadow-lg p-6 space-y-4">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-2">
              Название *
            </label>
            <input
              type="text"
              id="title"
              name="title"
              required
              value={formData.title}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Введите название видео"
            />
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
              Описание
            </label>
            <textarea
              id="description"
              name="description"
              rows="4"
              value={formData.description}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Введите описание видео"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="course_name" className="block text-sm font-medium text-gray-700 mb-2">
                Название курса
              </label>
              <input
                type="text"
                id="course_name"
                name="course_name"
                value={formData.course_name}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="например, Математика 10 класс"
              />
            </div>

            <div>
              <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                Предмет
              </label>
              <input
                type="text"
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                placeholder="например, Алгебра"
              />
            </div>
          </div>
        </div>

        {/* Upload Progress */}
        {uploading && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="mb-2 flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">Загрузка...</span>
              <span className="text-sm font-semibold text-primary-600">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div
                className="bg-primary-600 h-3 transition-all duration-300 rounded-full"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* Submit Buttons */}
        <div className="flex space-x-4">
          <button
            type="submit"
            disabled={uploading || !file}
            className="flex-1 py-3 px-6 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold"
          >
            {uploading ? 'Загрузка...' : 'Загрузить видео'}
          </button>
          <button
            type="button"
            onClick={() => navigate('/videos')}
            disabled={uploading}
            className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors font-semibold"
          >
            Отмена
          </button>
        </div>
      </form>
    </div>
  );
};

export default VideoUpload;
