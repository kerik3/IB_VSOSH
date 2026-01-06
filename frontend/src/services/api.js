import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Не перенаправляем на login при ошибках на странице входа/регистрации
    const isAuthPage = window.location.pathname === '/login' || window.location.pathname === '/register';
    
    if (error.response?.status === 401 && !isAuthPage) {
      // Только очищаем токен, редирект произойдёт через ProtectedRoute
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (credentials) => api.post('/auth/login', credentials),
  register: (userData) => api.post('/auth/register', userData),
  getCurrentUser: () => api.get('/auth/me'),
};

// Video API
export const videoAPI = {
  list: () => api.get('/videos'),
  get: (id) => api.get(`/videos/${id}`),
  upload: (formData, onProgress) => 
    api.post('/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress,
    }),
  update: (id, data) => api.put(`/videos/${id}`, data),
  delete: (id) => api.delete(`/videos/${id}`),
  getStreamUrl: (id) => `${API_BASE_URL}/videos/${id}/stream`,
};

// Access API
export const accessAPI = {
  getVideoAccess: (videoId) => api.get(`/videos/${videoId}/access`),
  grantAccess: (videoId, studentIds) => 
    api.post(`/videos/${videoId}/access`, { student_ids: studentIds }),
  revokeAccess: (videoId, studentId) => 
    api.delete(`/videos/${videoId}/access/${studentId}`),
};

// Leak detection API
export const leakAPI = {
  detect: (formData) => 
    api.post('/leaks/detect', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
  list: () => api.get('/leaks'),
  update: (id, data) => api.put(`/leaks/${id}`, data),
};

// User management API
export const userAPI = {
  list: (role) => api.get('/users', { params: { role } }),
  ban: (id, reason) => api.post(`/users/${id}/ban`, { reason }),
  unban: (id) => api.post(`/users/${id}/unban`),
};

// Statistics API
export const statsAPI = {
  get: () => api.get('/stats'),
};

export default api;
