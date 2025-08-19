import axios from 'axios';

const API_BASE_URL = 'http://localhost:8008';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
export const setAuthToken = (token: string) => {
  api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
};

export const clearAuthToken = () => {
  delete api.defaults.headers.common['Authorization'];
};

// Auth API
export const authAPI = {
  login: async (username: string, password: string) => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },
  
  logout: async () => {
    const response = await api.post('/auth/logout');
    return response.data;
  },
  
  getMe: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },
  
  refreshToken: async (refreshToken: string) => {
    const response = await api.post('/auth/refresh', { refresh_token: refreshToken });
    return response.data;
  }
};

// Chat API
export const chatAPI = {
  sendMessage: async (message: string) => {
    const response = await api.post('/chat', { message });
    return response.data;
  }
};

// Upload API
export const uploadAPI = {
  uploadFile: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }
};

// Search API
export const searchAPI = {
  search: async (query: string) => {
    const response = await api.post('/search', { query });
    return response.data;
  }
};

// Agents API
export const agentsAPI = {
  getAgents: async () => {
    const response = await api.get('/agents');
    return response.data;
  }
};

// Documents API
export const documentsAPI = {
  getDocuments: async () => {
    const response = await api.get('/documents');
    return response.data;
  }
};

// Health API
export const healthAPI = {
  checkHealth: async () => {
    const response = await api.get('/admin/health');
    return response.data;
  }
};

export default api; 