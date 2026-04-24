import axios from 'axios';

// Local development API base URL
const API_BASE_URL = 'http://localhost:8000';

/**
 * Upload image and get dog detection + emotion classification results
 * @param {File} imageFile - The image file to upload
 * @returns {Promise<Object>} Detection results
 */
export const detectEmotion = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await axios.post(`${API_BASE_URL}/api/detect`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    if (error.response) {
      // Server responded with error status
      throw new Error(error.response.data.detail || 'Detection failed');
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Cannot connect to server. Is the API running?');
    } else {
      // Something else happened
      throw new Error('An error occurred during detection');
    }
  }
};

/**
 * Check API health status
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    throw new Error('Cannot connect to API server');
  }
};
