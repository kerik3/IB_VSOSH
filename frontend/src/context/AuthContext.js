import React, { createContext, useState, useContext, useEffect } from 'react';
import { authAPI } from '../services/api';
import { toast } from 'react-toastify';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // При загрузке приложения проверяем есть ли сохранённый пользователь
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = localStorage.getItem('token');
      const savedUser = localStorage.getItem('user');
      
      console.log('AuthContext init:', { hasToken: !!savedToken, hasUser: !!savedUser });
      
      if (savedToken && savedUser) {
        try {
          // Сначала пробуем использовать сохранённого пользователя
          const parsedUser = JSON.parse(savedUser);
          setUser(parsedUser);
          console.log('User restored from localStorage:', parsedUser.username);
          
          // Затем пробуем обновить данные с сервера (но не блокируем если не получится)
          try {
            const response = await authAPI.getCurrentUser();
            if (response.data.user) {
              setUser(response.data.user);
              localStorage.setItem('user', JSON.stringify(response.data.user));
              console.log('User data refreshed from server');
            }
          } catch (refreshError) {
            console.warn('Could not refresh user from server:', refreshError.message);
            // Не очищаем данные - продолжаем с сохранёнными
          }
        } catch (parseError) {
          console.error('Failed to parse saved user:', parseError);
          localStorage.removeItem('user');
        }
      }
      
      setLoading(false);
    };
    
    initAuth();
  }, []);

  const login = async (username, password) => {
    try {
      console.log('Attempting login for:', username);
      const response = await authAPI.login({ username, password });
      const { access_token, user: userData } = response.data;
      
      console.log('Login successful, saving token and user');
      
      // Сохраняем в localStorage
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(userData));
      
      // Обновляем state
      setUser(userData);
      
      toast.success(`Добро пожаловать, ${userData.full_name}!`);
      return { success: true, user: userData };
    } catch (error) {
      console.error('Login failed:', error.response?.data || error.message);
      const message = error.response?.data?.error || 'Ошибка входа';
      toast.error(message);
      return { success: false, error: message };
    }
  };

  const register = async (userData) => {
    try {
      await authAPI.register(userData);
      toast.success('Регистрация успешна! Теперь войдите в аккаунт.');
      return true;
    } catch (error) {
      const message = error.response?.data?.error || 'Ошибка регистрации';
      toast.error(message);
      return false;
    }
  };

  const logout = () => {
    console.log('Logging out');
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    toast.info('Вы вышли из аккаунта');
  };

  const value = {
    user,
    login,
    register,
    logout,
    loading,
    isAuthenticated: !!user,
    isTeacher: user?.role === 'teacher',
    isStudent: user?.role === 'student',
    isAdmin: user?.role === 'admin',
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
