import  { useState } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('login');

  const handleTabClick = (tab) => {
    setActiveTab(tab);
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
        {activeTab === 'login' && (
          <div className="w-full">
            <p className="text-center mb-3 ">Sign in with:</p>
            <div className="flex justify-between mb-5 w-1/2 mx-auto">
              <button className="text-blue-600">
                <i className="fab fa-facebook-f"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-twitter"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-google"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-github"></i>
              </button>
            </div>

            <p className="text-center mb-3">or:</p>
            <input type="email" placeholder="Email address" className="mb-4 p-3 w-full border border-gray-300 rounded" />
            <input type="password" placeholder="Password" className="mb-4 p-3 w-full border border-gray-300 rounded" />

            <div className="flex justify-between mb-4">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" /> Remember me
              </label>
              <a href="#!" className="text-blue-500">Forgot password?</a>
            </div>

            <button className="w-full p-3 bg-blue-500 text-white rounded">Sign in</button>
            <p className="text-center mt-3">Not a member? <a href="#!" className="text-blue-500">Register</a></p>
          </div>
        )}

        {activeTab === 'register' && (
          <div className="w-full">
            <p className="text-center mb-3">Sign up with:</p>
            <div className="flex justify-between mb-5 w-1/2 mx-auto">
              <button className="text-blue-600">
                <i className="fab fa-facebook-f"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-twitter"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-google"></i>
              </button>
              <button className="text-blue-600">
                <i className="fab fa-github"></i>
              </button>
            </div>

            <p className="text-center mb-3">or:</p>
            <input type="text" placeholder="Name" className="mb-4 p-3 w-full border border-gray-300 rounded" />
            <input type="text" placeholder="Mobile No." className="mb-4 p-3 w-full border border-gray-300 rounded" />
            <input type="email" placeholder="Email" className="mb-4 p-3 w-full border border-gray-300 rounded" />
            <input type="password" placeholder="Password" className="mb-4 p-3 w-full border border-gray-300 rounded" />

            <div className="flex justify-center mb-4">
              <label className="flex items-center">
                <input type="checkbox" className="mr-2" /> I have read and agree to the terms
              </label>
            </div>

            <button className="w-full p-3 bg-blue-500 text-white rounded">Sign up</button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
