import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { statsAPI } from '../services/api';
import { FiVideo, FiUsers, FiAlertTriangle, FiShield } from 'react-icons/fi';

const StatCard = ({ icon: Icon, title, value, color }) => (
  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-500 text-sm font-medium uppercase">{title}</p>
        <p className="text-3xl font-bold text-gray-900 mt-2">{value ?? 0}</p>
      </div>
      <div className={`p-4 rounded-full ${color}`}>
        <Icon className="w-8 h-8 text-white" />
      </div>
    </div>
  </div>
);

const Dashboard = () => {
  const { user, isTeacher, isStudent, isAdmin } = useAuth();
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await statsAPI.get();
        setStats(response.data.stats || {});
      } catch (error) {
        console.error('Failed to fetch statistics:', error);
        // Не показываем ошибку пользователю, просто оставляем пустую статистику
        setStats({});
      } finally {
        setLoading(false);
      }
    };
    
    fetchStats();
  }, []);

  const renderAdminStats = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        icon={FiUsers}
        title="Всего пользователей"
        value={stats.total_users}
        color="bg-blue-500"
      />
      <StatCard
        icon={FiVideo}
        title="Всего видео"
        value={stats.total_videos}
        color="bg-green-500"
      />
      <StatCard
        icon={FiShield}
        title="С водяным знаком"
        value={stats.total_watermarked}
        color="bg-purple-500"
      />
      <StatCard
        icon={FiAlertTriangle}
        title="Утечки"
        value={stats.total_leaks}
        color="bg-red-500"
      />
      <StatCard
        icon={FiUsers}
        title="Учеников"
        value={stats.total_students}
        color="bg-indigo-500"
      />
      <StatCard
        icon={FiUsers}
        title="Преподавателей"
        value={stats.total_teachers}
        color="bg-cyan-500"
      />
      <StatCard
        icon={FiAlertTriangle}
        title="Ожидают проверки"
        value={stats.pending_leaks}
        color="bg-yellow-500"
      />
      <StatCard
        icon={FiAlertTriangle}
        title="Подтверждённые утечки"
        value={stats.confirmed_leaks}
        color="bg-red-600"
      />
    </div>
  );

  const renderTeacherStats = () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatCard
        icon={FiVideo}
        title="Загружено видео"
        value={stats.uploaded_videos}
        color="bg-blue-500"
      />
      <StatCard
        icon={FiUsers}
        title="Учеников"
        value={stats.total_students}
        color="bg-green-500"
      />
      <StatCard
        icon={FiShield}
        title="Просмотров"
        value={stats.total_views}
        color="bg-purple-500"
      />
    </div>
  );

  const renderStudentStats = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <StatCard
        icon={FiVideo}
        title="Доступно видео"
        value={stats.accessible_videos}
        color="bg-blue-500"
      />
      <StatCard
        icon={FiShield}
        title="Просмотрено"
        value={stats.watched_videos}
        color="bg-green-500"
      />
    </div>
  );

  // Если нет пользователя - ничего не показываем (ProtectedRoute разберётся)
  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          С возвращением, {user.full_name}!
        </h1>
        <p className="text-gray-600 mt-2">
          {isAdmin && "Управляйте платформой из этой панели"}
          {isTeacher && "Управляйте своими видео и учениками"}
          {isStudent && "Доступ к учебным материалам"}
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div>
          {isAdmin && renderAdminStats()}
          {isTeacher && renderTeacherStats()}
          {isStudent && renderStudentStats()}
        </div>
      )}

      <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Быстрые действия</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(isTeacher || isAdmin) && (
            <>
              <a
                href="/videos"
                className="p-4 border-2 border-primary-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
              >
                <FiVideo className="w-8 h-8 mx-auto text-primary-600 mb-2" />
                <h3 className="font-semibold text-gray-900">Управление видео</h3>
                <p className="text-sm text-gray-600 mt-1">Загружайте и управляйте контентом</p>
              </a>
              
              <a
                href="/leak-detection"
                className="p-4 border-2 border-red-200 rounded-lg hover:border-red-500 hover:bg-red-50 transition-colors text-center"
              >
                <FiAlertTriangle className="w-8 h-8 mx-auto text-red-600 mb-2" />
                <h3 className="font-semibold text-gray-900">Детекция утечек</h3>
                <p className="text-sm text-gray-600 mt-1">Обнаружение и отчёты об утечках</p>
              </a>
            </>
          )}
          
          {isStudent && (
            <a
              href="/videos"
              className="p-4 border-2 border-primary-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-center"
            >
              <FiVideo className="w-8 h-8 mx-auto text-primary-600 mb-2" />
              <h3 className="font-semibold text-gray-900">Мои видео</h3>
              <p className="text-sm text-gray-600 mt-1">Доступ к учебным материалам</p>
            </a>
          )}
          
          {isAdmin && (
            <a
              href="/users"
              className="p-4 border-2 border-green-200 rounded-lg hover:border-green-500 hover:bg-green-50 transition-colors text-center"
            >
              <FiUsers className="w-8 h-8 mx-auto text-green-600 mb-2" />
              <h3 className="font-semibold text-gray-900">Пользователи</h3>
              <p className="text-sm text-gray-600 mt-1">Управление пользователями</p>
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
