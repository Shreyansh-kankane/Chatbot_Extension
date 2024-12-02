import React from 'react';
import { Navigate } from 'react-router-dom';

interface AuthProps {
    children: React.ReactNode;
}

const RedirectIfAuthenticated: React.FC<AuthProps> = ({ children }) => {
    const authToken = sessionStorage.getItem('authToken');
    if (authToken) {
        return <Navigate to="/" replace />;
    }
    return <>{children}</>;
};

export default RedirectIfAuthenticated;
