const authFetch = async (url, options = {}) => {
    const authToken = sessionStorage.getItem('authToken');
    const headers = {
      'Authorization': `Bearer ${authToken}`,
      'Content-Type': 'application/json',
      ...options.headers,
    };
  
    return fetch(url, { ...options, headers });
  };