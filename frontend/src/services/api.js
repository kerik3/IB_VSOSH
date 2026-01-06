import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add token to requests and adjust headers for FormData
api.interceptors.request.use(
  (config) => {
    // Remove default content-type for FormData so browser sets boundary
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }

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
    const status = error.response?.status;
    const isAuthPage = window.location.pathname === '/login' || window.location.pathname === '/register';

    // JWT ошибки могут приходить как 401 или 422 (невалидный/просроченный токен)
    if ((status === 401 || status === 422) && !isAuthPage) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      // Дальше ProtectedRoute отправит на /login
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
    api.post('/leaks/detect', formData),
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
