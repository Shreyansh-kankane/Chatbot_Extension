// AppRoutes.js
import { Routes, Route } from 'react-router-dom';
import Onboarding from '../pages/onboarding';
import LandingPage from '../pages/landingPage';

function AppRoutes() {
  return (
    <Routes>
      <Route path='/'  element={<LandingPage/>}/>
      <Route path="/onboarding/login" element={<div><Onboarding /></div>} />
      <Route path="/onboarding/register" element={<Onboarding />} />
    </Routes>
  );
}

export default AppRoutes;
