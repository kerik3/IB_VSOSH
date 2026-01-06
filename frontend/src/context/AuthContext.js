import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
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
  // Инициализируем user из localStorage если есть
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('user');
    return savedUser ? JSON.parse(savedUser) : null;
  });
  const [loading, setLoading] = useState(() => {
    // Если есть сохранённый пользователь, не показываем загрузку
    return !localStorage.getItem('user') && !!localStorage.getItem('token');
  });

  // Проверяем токен при первой загрузке (если есть токен, но нет user)
  useEffect(() => {
    const token = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (token && !savedUser) {
      fetchCurrentUser();
    }
  }, []);

  const fetchCurrentUser = async () => {
    try {
      const response = await authAPI.getCurrentUser();
      const userData = response.data.user;
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    } catch (error) {
      console.error('Не удалось получить данные пользователя:', error);
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = useCallback(async (username, password) => {
    try {
      const response = await authAPI.login({ username, password });
      const { access_token, user: userData } = response.data;
      
      // Сохраняем в localStorage
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(userData));
      
      // Обновляем state
      setUser(userData);
      setLoading(false);
      
      toast.success(`Добро пожаловать, ${userData.full_name}!`);
      return { success: true, user: userData };
    } catch (error) {
      const message = error.response?.data?.error || 'Ошибка входа';
      toast.error(message);
      return { success: false };
    }
  }, []);

  const register = useCallback(async (userData) => {
    try {
      await authAPI.register(userData);
      toast.success('Регистрация успешна! Теперь войдите в аккаунт.');
      return true;
    } catch (error) {
      const message = error.response?.data?.error || 'Ошибка регистрации';
      toast.error(message);
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    toast.info('Вы успешно вышли из аккаунта');
  }, []);

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
