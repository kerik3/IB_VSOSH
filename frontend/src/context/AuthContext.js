import React, { createContext, useState, useContext, useEffect, useRef } from 'react';
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
  const [token, setToken] = useState(localStorage.getItem('token'));
  const isInitialMount = useRef(true);

  useEffect(() => {
    // Только при первой загрузке проверяем токен
    if (isInitialMount.current) {
      isInitialMount.current = false;
      if (token) {
        fetchCurrentUser();
      } else {
        setLoading(false);
      }
    }
  }, []);

  const fetchCurrentUser = async () => {
    try {
      const response = await authAPI.getCurrentUser();
      setUser(response.data.user);
    } catch (error) {
      console.error('Не удалось получить данные пользователя:', error);
      // Тихо очищаем данные без показа уведомления (токен просто истёк)
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (username, password) => {
    try {
      const response = await authAPI.login({ username, password });
      const { access_token, user: userData } = response.data;
      
      localStorage.setItem('token', access_token);
      setToken(access_token);
      setUser(userData);
      setLoading(false);
      
      toast.success(`Добро пожаловать, ${userData.full_name}!`);
      return { success: true, user: userData };
    } catch (error) {
      const message = error.response?.data?.error || 'Ошибка входа';
      toast.error(message);
      return { success: false };
    }
  };

  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);
      toast.success('Регистрация успешна! Теперь войдите в аккаунт.');
      return true;
    } catch (error) {
      const message = error.response?.data?.error || 'Ошибка регистрации';
      toast.error(message);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
    toast.info('Вы успешно вышли из аккаунта');
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
