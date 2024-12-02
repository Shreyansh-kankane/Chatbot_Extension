import React from 'react';
import { Navigate } from 'react-router-dom';

interface AuthProps {
    children: React.ReactNode;
}

const AuthMiddleware: React.FC<AuthProps> = ({ children }) => {
    const authToken = sessionStorage.getItem('authToken');
    
    // If no auth token, redirect to login
    if (!authToken) {
        return <Navigate to="/auth" replace />;
    }

    return <>{children}</>;
};

export default AuthMiddleware;
