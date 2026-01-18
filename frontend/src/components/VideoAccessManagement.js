import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { FiArrowLeft, FiUserPlus, FiUserMinus, FiSearch, FiCheck, FiX, FiUsers } from 'react-icons/fi';
import { videoAPI, accessAPI, userAPI } from '../services/api';

const VideoAccessManagement = () => {
    const { id } = useParams();
    const navigate = useNavigate();

    const [video, setVideo] = useState(null);
    const [students, setStudents] = useState([]); // All students
    const [accessList, setAccessList] = useState([]); // Students with access
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedStudents, setSelectedStudents] = useState([]);

    useEffect(() => {
        fetchData();
    }, [id]);

    const fetchData = async () => {
        try {
            setLoading(true);
            // Run requests in parallel
            const [videoRes, accessRes, studentsRes] = await Promise.all([
                videoAPI.get(id),
                accessAPI.getVideoAccess(id),
                userAPI.list('student')
            ]);

            setVideo(videoRes.data.video);
            setAccessList(accessRes.data.access_list || []);
            setStudents(studentsRes.data.users);

        } catch (error) {
            console.error('Error fetching data:', error);
            toast.error('Ошибка загрузки данных');
            // If video invalid, go back
            if (!video) navigate('/videos');
        } finally {
            setLoading(false);
        }
    };

    const handleGrantAccess = async () => {
        if (selectedStudents.length === 0) return;

        try {
            await accessAPI.grantAccess(id, selectedStudents);
            toast.success('Доступ предоставлен');
            setSelectedStudents([]);
            // Refresh access list
            const accessRes = await accessAPI.getVideoAccess(id);
            setAccessList(accessRes.data.access_list || []);
        } catch (error) {
            console.error('Error granting access:', error);
            toast.error('Ошибка предоставления доступа');
        }
    };

    const handleRevokeAccess = async (studentId) => {
        if (!window.confirm('Вы уверены, что хотите закрыть доступ для этого студента?')) {
            return;
        }

        try {
            await accessAPI.revokeAccess(id, studentId);
            toast.success('Доступ закрыт');
            // Refresh access list
            const accessRes = await accessAPI.getVideoAccess(id);
            setAccessList(accessRes.data.access_list || []);
        } catch (error) {
            console.error('Error revoking access:', error);
            toast.error('Ошибка отмены доступа');
        }
    };

    const toggleStudentSelection = (studentId) => {
        setSelectedStudents(prev =>
            prev.includes(studentId)
                ? prev.filter(id => id !== studentId)
                : [...prev, studentId]
        );
    };

    // Filter students who don't have access yet
    const studentsWithoutAccess = students.filter(student =>
        !accessList.some(item => item.student_id === student.id)
    ).filter(student =>
        student.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.full_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.email?.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
            </div>
        );
    }

    if (!video) return null;

    return (
        <div className="p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <button
                        onClick={() => navigate('/videos')}
                        className="flex items-center space-x-2 text-gray-600 hover:text-primary-600 transition-colors mb-2"
                    >
                        <FiArrowLeft className="w-5 h-5" />
                        <span>Назад к списку</span>
                    </button>
                    <h1 className="text-3xl font-bold text-gray-900">Управление доступом</h1>
                    <p className="text-gray-600 mt-1">Видео: <span className="font-semibold">{video.title}</span></p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column: Grant Access */}
                <div className="bg-white rounded-xl shadow-md border border-gray-100 p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                            <FiUserPlus className="text-primary-600" />
                            Добавить студентов
                        </h2>
                        {selectedStudents.length > 0 && (
                            <button
                                onClick={handleGrantAccess}
                                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
                            >
                                Добавить ({selectedStudents.length})
                            </button>
                        )}
                    </div>

                    <div className="relative mb-4">
                        <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Поиск студентов..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        />
                    </div>

                    <div className="overflow-y-auto max-h-[500px] space-y-2">
                        {studentsWithoutAccess.length === 0 ? (
                            <p className="text-gray-500 text-center py-8">
                                {searchTerm ? 'Ничего не найдено' : 'Все студенты уже имеют доступ'}
                            </p>
                        ) : (
                            studentsWithoutAccess.map(student => (
                                <div
                                    key={student.id}
                                    onClick={() => toggleStudentSelection(student.id)}
                                    className={`flex items-center justify-between p-3 rounded-lg cursor-pointer border transition-all ${selectedStudents.includes(student.id)
                                            ? 'bg-primary-50 border-primary-200 ring-1 ring-primary-500'
                                            : 'bg-gray-50 border-transparent hover:bg-gray-100'
                                        }`}
                                >
                                    <div>
                                        <p className="font-medium text-gray-900">{student.full_name || student.username}</p>
                                        <p className="text-sm text-gray-500">{student.email}</p>
                                    </div>
                                    <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${selectedStudents.includes(student.id)
                                            ? 'bg-primary-600 border-primary-600'
                                            : 'border-gray-300'
                                        }`}>
                                        {selectedStudents.includes(student.id) && <FiCheck className="text-white w-4 h-4" />}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Right Column: Current Access */}
                <div className="bg-white rounded-xl shadow-md border border-gray-100 p-6">
                    <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                        <FiUsers className="text-green-600" />
                        Текущий доступ ({accessList.length})
                    </h2>

                    <div className="overflow-y-auto max-h-[600px] space-y-2">
                        {accessList.length === 0 ? (
                            <p className="text-gray-500 text-center py-8">
                                Доступ пока никому не предоставлен
                            </p>
                        ) : (
                            accessList.map(item => (
                                <div
                                    key={item.student_id}
                                    className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg hover:shadow-sm transition-shadow"
                                >
                                    <div>
                                        <p className="font-medium text-gray-900">{item.student_name || item.student_username}</p>
                                        <div className="flex items-center gap-2 text-sm text-gray-500">
                                            <span>Выдан: {new Date(item.granted_at).toLocaleDateString()}</span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => handleRevokeAccess(item.student_id)}
                                        className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                                        title="Закрыть доступ"
                                    >
                                        <FiUserMinus className="w-5 h-5" />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VideoAccessManagement;
