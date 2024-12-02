import { useState } from 'react';
import { useNavigate } from 'react-router-dom';


function Authenticate() {
  
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('login');
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [registerData, setRegisterData] = useState({
    name: '',
    mobile: '',
    email: '',
    password: '',
  });
  
  const handleTabClick = (tab) => {
    setActiveTab(tab);
  };
  
  const handleInputChange = (e, type) => {
    const { name, value } = e.target;
    if (type === 'login') {
      setLoginData((prev) => ({ ...prev, [name]: value }));
    } else {
      setRegisterData((prev) => ({ ...prev, [name]: value }));
    }
  };
  
  const handleLogin = async (e) => {
    const backendUri = process.env.BACKEND_URI;
    e.preventDefault();
    try {
      const response = await fetch(`${backendUri}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginData),
      });

      const result = await response.json();

      if (response.ok) {
        sessionStorage.setItem('authToken', result.token);
        navigate('/dashboard')

      } else {
        alert(data.message || 'Login failed');
      }
      
    } catch (error) {
      console.error('Login error:', error);
    }
  };

  const handleRegister = async (e) => {
    const backendUri = process.env.BACKEND_URI;
    e.preventDefault();
    try {
      const response = await fetch(`${backendUri}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(registerData),
      });

      const result = await response.json();
      console.log(result); // handle register response

      if (response.ok) {
        navigate('/dashboard');
      } else {
        alert(data.message || 'Login failed');
      }
      
    } catch (error) {
      console.error('Register error:', error);
    }
  };

  return (
    <div className="p-8 my-10 flex flex-col w-full max-w-lg mx-auto bg-[#16141A] text-white shadow-lg rounded-lg">
      {/* Tabs */}
      <div className="flex justify-between mb-5 border-b border-gray-200 font-bold text-lg">
        <button
          onClick={() => handleTabClick('login')}
          className={`pb-2 w-1/2 text-center ${activeTab === 'login' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-white'}`}
        >
          Login
        </button>
        <button
          onClick={() => handleTabClick('register')}
          className={`pb-2 w-1/2 text-center ${activeTab === 'register' ? 'border-b-2 border-blue-500 text-blue-500' : 'text-white'}`}
        >
          Register
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex flex-col items-center">
        {activeTab === 'login' ? (
          <form className="w-full" onSubmit={handleLogin}>
            <p className="text-center mb-3">Sign in with:</p>
            <div className="flex justify-between mb-5 w-1/2 mx-auto">
              <button className="text-blue-600"><i className="fab fa-facebook-f"></i></button>
              <button className="text-blue-600"><i className="fab fa-twitter"></i></button>
              <button className="text-blue-600"><i className="fab fa-google"></i></button>
              <button className="text-blue-600"><i className="fab fa-github"></i></button>
            </div>
            <p className="text-center mb-3">or:</p>
            <input
              type="email"
              name="email"
              placeholder="Email address"
              className="mb-4 p-3 w-full border border-gray-300 text-black rounded"
              value={loginData.email}
              onChange={(e) => handleInputChange(e, 'login')}
              required
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="mb-4 p-3 w-full border border-gray-300 text-black rounded"
              value={loginData.password}
              onChange={(e) => handleInputChange(e, 'login')}
              required
            />
            <div className="flex justify-between mb-4">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" /> Remember me
              </label>
              <a href="#!" className="text-blue-500">Forgot password?</a>
            </div>
            <button type="submit" className="w-full p-3 bg-blue-500 text-white rounded">Sign in</button>
            <p className="text-center mt-3">Not a member? <a href="#!" className="text-blue-500">Register</a></p>
          </form>
        ) : (
          <form className="w-full" onSubmit={handleRegister}>
            <p className="text-center mb-3">Sign up with:</p>
            <div className="flex justify-between mb-5 w-1/2 mx-auto">
              <button className="text-blue-600"><i className="fab fa-facebook-f"></i></button>
              <button className="text-blue-600"><i className="fab fa-twitter"></i></button>
              <button className="text-blue-600"><i className="fab fa-google"></i></button>
              <button className="text-blue-600"><i className="fab fa-github"></i></button>
            </div>
            <p className="text-center mb-3">or:</p>
            <input
              type="text"
              name="name"
              placeholder="Name"
              className="mb-4 p-3 w-full border border-gray-300 text-black rounded"
              value={registerData.name}
              onChange={(e) => handleInputChange(e, 'register')}
              required
            />
            <input
              type="text"
              name="mobile"
              placeholder="Mobile No."
              className="mb-4 p-3 w-full border border-gray-300 text-black rounded"
              value={registerData.mobile}
              onChange={(e) => handleInputChange(e, 'register')}
              required
            />
            <input
              type="email"
              name="email"
              placeholder="Email"
              className="mb-4 p-3 w-full border border-gray-300 text-black  rounded"
              value={registerData.email}
              onChange={(e) => handleInputChange(e, 'register')}
              required
            />
            <input
              type="password"
              name="password"
              placeholder="Password"
              className="mb-4 p-3 w-full border border-gray-300 text-black rounded"
              value={registerData.password}
              onChange={(e) => handleInputChange(e, 'register')}
              required
            />
            <div className="flex justify-center mb-4">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" required /> I have read and agree to the terms
              </label>
            </div>
            <button type="submit" className="w-full p-3 bg-blue-500 text-white rounded">Sign up</button>
          </form>
        )}
      </div>
    </div>
  );
}

export default Authenticate;
