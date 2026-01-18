import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { videoAPI } from '../services/api';
import { toast } from 'react-toastify';
import { FiSave, FiX, FiArrowLeft } from 'react-icons/fi';

const VideoEdit = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    const [formData, setFormData] = useState({
        title: '',
        description: '',
        course_name: '',
        subject: '',
        is_active: true
    });

    useEffect(() => {
        fetchVideo();
    }, [id]);

    const fetchVideo = async () => {
        try {
            const response = await videoAPI.get(id);
            const video = response.data.video;
            setFormData({
                title: video.title || '',
                description: video.description || '',
                course_name: video.course_name || '',
                subject: video.subject || '',
                is_active: video.is_active
            });
        } catch (error) {
            toast.error('Ошибка загрузки данных видео');
            navigate('/videos');
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e) => {
        const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
        setFormData({
            ...formData,
            [e.target.name]: value,
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setSaving(true);

        try {
            await videoAPI.update(id, formData);
            toast.success('Видео успешно обновлено');
            navigate('/videos');
        } catch (error) {
            console.error('Update error:', error);
            toast.error('Ошибка обновления видео');
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
            </div>
        );
    }

    return (
        <div className="p-6 max-w-4xl mx-auto">
            <div className="mb-8 flex items-center justify-between">
                <div>
                    <button
                        onClick={() => navigate('/videos')}
                        className="flex items-center space-x-2 text-gray-600 hover:text-primary-600 transition-colors mb-2"
                    >
                        <FiArrowLeft className="w-5 h-5" />
                        <span>Назад к списку</span>
                    </button>
                    <h1 className="text-3xl font-bold text-gray-900">Редактирование видео</h1>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
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
                            />
                        </div>
                    </div>

                    <div className="flex items-center space-x-3 pt-4 border-t border-gray-100">
                        <input
                            type="checkbox"
                            id="is_active"
                            name="is_active"
                            checked={formData.is_active}
                            onChange={handleChange}
                            className="h-5 w-5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                        />
                        <label htmlFor="is_active" className="text-sm font-medium text-gray-700 select-none cursor-pointer">
                            Видео активно (видно студентам с доступом)
                        </label>
                    </div>
                </div>

                <div className="flex space-x-4">
                    <button
                        type="submit"
                        disabled={saving}
                        className="flex-1 py-3 px-6 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors font-semibold flex items-center justify-center space-x-2"
                    >
                        <FiSave className="w-5 h-5" />
                        <span>{saving ? 'Сохранение...' : 'Сохранить изменения'}</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => navigate('/videos')}
                        disabled={saving}
                        className="px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors font-semibold flex items-center space-x-2"
                    >
                        <FiX className="w-5 h-5" />
                        <span>Отмена</span>
                    </button>
                </div>
            </form>
        </div>
    );
};

export default VideoEdit;
